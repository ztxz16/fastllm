#include "tfacc2.h"
#include "linux/of.h"
#include <asm/pgtable.h>
#include <linux/time.h>
#include <linux/jiffies.h>

static struct tf_device *tf_dev = NULL;
static bool versionIsRight = false; // 表明version是否check过，如果check成功则置为true
static int isNextUncache = 0;
static int chips = 1;
static int efuse = 0;
static int chipInit = 0;
static int cacheInit = 0;
static int chipInitCnt = 0;
static int cacheInitCnt = 0;
static int skipFullSwRst = 0;
static unsigned long long chipGap = 0; // 相邻两个chip之间的寄存器地址差多少
static unsigned long long ddrStart = 0x100000000LL;

static int needReset = 0;

static void cpu_write(unsigned long long base, int value) {
    volatile unsigned int *clk = ioremap(base, 4);
    clk[0] = value;
    iounmap(clk);
}

static void hardware_tfacc_reset(void) {
    if (efuse) {
        return;
    }
    // 下面这段代码是硬件复位
    cpu_write(0xfe110000 + 0x84 * 4, 0x2);
    cpu_write(0xfc200000 + 0xe0, 0x0);
    cpu_write(0xfc200000 + 0xf8, 0x0);
    cpu_write(0xfc200000 + 0xfc, 0x0);
    cpu_write(0xfc600000 + 0xe0, 0x0);
    cpu_write(0xfc600000 + 0xf8, 0x0);
    cpu_write(0xfc600000 + 0xfc, 0x0);
    cpu_write(0xfe110000 + 0x82 * 4, 0x1);
    cpu_write(0xfe110000 + 0x82 * 4, 0x3);
    cpu_write(0xfe110000 + 0x84 * 4, 0x0);

    cpu_write(0xed400000 + 0x84 * 4, 0x3);
    cpu_write(0xec200000 + 0xe0, 0x0);
    cpu_write(0xec200000 + 0xf8, 0x0);
    cpu_write(0xec200000 + 0xfc, 0x0);
    cpu_write(0xec600000 + 0xe0, 0x0);
    cpu_write(0xec600000 + 0xf8, 0x0);
    cpu_write(0xec600000 + 0xfc, 0x0);
    cpu_write(0xed400000 + 0x82 * 4, 0x1);
    cpu_write(0xed400000 + 0x82 * 4, 0x3);
    cpu_write(0xed400000 + 0x84 * 4, 0x1);
}

/* reserve mem */
struct ReserveDDRBlock reserveDDRBlocks[1005];
int reserveDDRBlcokCnt = 0;

static int tf_get_reserve_ddr_blocks(struct ReserveDDRBlock* p) {
    int i;
    int ret;
    if (efuse) {
        return -1;
    }

    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        ret = copy_to_user((void *) p, &reserveDDRBlocks[i], sizeof(struct ReserveDDRBlock));
        p++;
    }
    return 0;
}

/* app lock */
static struct mutex app_mutex;
static spinlock_t app_spin_lock;

#define MAX_APP_LOCK_RECORDS 500
// Parameter 'size' for an ioctl code is limited with (16K -1).
static struct tf_lock_record app_lock_records[MAX_APP_LOCK_RECORDS];
static int curRecordOffset = 0;
static int tfaccRecordOffsets[MAX_TFACC_CNT];

// 记录一个TFACC的使用情况
// 分为两部分
// 第一部分是一个循环队列，记录过去最多300秒之内，每一秒的占用率
// 第二部分是还没有进入队列的信息
// 这里面事件记录均使用MS为单位
#define MAX_RECORD_SECONDS 300
#define RECORD_QUEUE_LEN 305
struct tf_use_record {
    // 这部分是循环队列, startTime[x]往后一秒内的使用率为percent[x] / 1000.0
    unsigned int percent[MAX_RECORD_SECONDS + 10];
    unsigned int startTime[MAX_RECORD_SECONDS + 10];
    int head, len;

    unsigned int lastTimeInQueue; // 即将计入队列的最早时间，这个数必须是1000的倍数。如果lastTimeInQueue = 2000，代表<=1999的数据都已经被计入队列了
    unsigned int lastRecordTime; // 最后一次记录的时间
    unsigned int lastUseTime; // 在[lastTimeInQueue, lastRecordTime]这段区间内有多久被锁定
    int isHolding;            // 是否还在持有
};
struct tf_use_record useRecords[MAX_TFACC_CNT];

static void initRecords(void) {
    int i = 0;
    if (efuse) {
        return;
    }
    for (i = 0; i < MAX_APP_LOCK_RECORDS; ++i) {
        app_lock_records[i].pid = 0;
        app_lock_records[i].tgid = 0;
        app_lock_records[i].isHolding = 0;
        app_lock_records[i].lockTime = 0;
        app_lock_records[i].unlockTime = 0;
        app_lock_records[i].tfaccID = -1;
    }

    for (i = 0; i < MAX_TFACC_CNT; ++i) {
        tfaccRecordOffsets[i] = 0;
    }

    for (i = 0; i < MAX_TFACC_CNT; i++) {
        useRecords[i].head = 0;
        useRecords[i].len = 0;
        useRecords[i].isHolding = 0;
        useRecords[i].lastTimeInQueue = 0;
        useRecords[i].lastRecordTime = 0;
        useRecords[i].lastUseTime = 0;
    }

    curRecordOffset = 0;
}

static void updateUseRecord(int tfaccID, int lastHolding, unsigned long curMs) {
    // 处理使用记录
    // 1. 先把超过300秒的记录都删掉
    struct tf_use_record *record = &useRecords[tfaccID];
    while (record->len > 0 && (curMs - record->startTime[record->head]) / 1000 + 1 > MAX_RECORD_SECONDS) {
        record->head = (record->head + 1) % RECORD_QUEUE_LEN;
        record->len--;
    }

    // 2. 处理LastRecordTime到当前时间
    if (record->isHolding) {
        // 如果最后处于锁定状态
        if ((curMs - record->lastTimeInQueue) / 1000 + 1 > MAX_RECORD_SECONDS) {
            // 上次记录已经过去很久了，调整到需要计入队列的最晚事件
            int x = (curMs / 1000 * 1000 - MAX_RECORD_SECONDS * 1000 + 1000);
            record->lastTimeInQueue = (x < 0) ? 0 : x;
            record->lastRecordTime = record->lastTimeInQueue;
            record->lastUseTime = 0;
        }

        while (record->lastTimeInQueue < curMs / 1000 * 1000) {
            int pos = (record->head + record->len) % RECORD_QUEUE_LEN;
            record->startTime[pos] = record->lastTimeInQueue;
            record->percent[pos] = (record->lastUseTime + (1000 - record->lastRecordTime % 1000));
            record->len++;

            record->lastTimeInQueue += 1000;
            record->lastRecordTime = record->lastTimeInQueue;
            record->lastUseTime = 0;
        }

        record->lastUseTime += (curMs - record->lastRecordTime);
        record->lastRecordTime = curMs;
    } else {
        // 如果最后处于非锁定状态
        if ((curMs - record->lastTimeInQueue) / 1000 + 1 > MAX_RECORD_SECONDS) {
            // 上次记录已经过去很久了.. 忽略
            record->lastTimeInQueue = curMs / 1000 * 1000;
            record->lastRecordTime = curMs;
            record->lastUseTime = 0;
        } else {
            // 上次记录的时间在300秒之内
            if (curMs / 1000 * 1000 == record->lastTimeInQueue) {
                // 上次记录还在1秒内
                record->lastRecordTime = curMs;
            } else {
                // 上次记录的时间超过1秒了，放入队列
                int pos = (record->head + record->len) % RECORD_QUEUE_LEN;
                record->startTime[pos] = record->lastTimeInQueue;
                record->percent[pos] = record->lastUseTime;
                record->len++;

                record->lastTimeInQueue = curMs / 1000 * 1000;
                record->lastRecordTime = curMs;
                record->lastUseTime = 0;
            }
        }
    }

    // 更新最终状态
    record->isHolding = lastHolding;
}

static void insertAppLockRecord(struct tf_device* dev, int tfaccID) {
    if (efuse) {
        return;
    }
    // 更新app 使用tfacc记录
    app_lock_records[curRecordOffset].pid = current->pid;
    app_lock_records[curRecordOffset].tgid = current->tgid;
    app_lock_records[curRecordOffset].lockTime = jiffies;
    app_lock_records[curRecordOffset].unlockTime = jiffies;
    app_lock_records[curRecordOffset].isHolding = 1;
    app_lock_records[curRecordOffset].tfaccID = tfaccID;

    dev->holdTFACCPid[tfaccID] = current->pid;
    dev->holdTFACCTgid[tfaccID] = current->tgid;

    tfaccRecordOffsets[tfaccID] = curRecordOffset;
    curRecordOffset = (curRecordOffset + 1) % MAX_APP_LOCK_RECORDS;

    updateUseRecord(tfaccID, 1, jiffies_to_msecs(jiffies));
}

static void finishAppLockRecord(struct tf_device* dev, int tfaccID) {
    int offset = tfaccRecordOffsets[tfaccID];
    if (efuse) {
        return;
    }

    dev->holdTFACCPid[tfaccID] = -1;
    dev->holdTFACCTgid[tfaccID] = -1;
    app_lock_records[offset].unlockTime = jiffies;
    app_lock_records[offset].isHolding = 0;

    // curRecordOffset = (curRecordOffset + 1) % MAX_APP_LOCK_RECORDS;
    updateUseRecord(tfaccID, 0, jiffies_to_msecs(jiffies));
}

static int tf_app_try_lock(struct tf_device* dev, int* p) {
    int sleepUs = 0;
    int tfaccID = -1;
    int lock_pid = -1;
    long uWait = (long) sleepUs;

    if (efuse) {
        return -1;
    }
    __get_user(sleepUs, p);
    p++;
    __get_user(tfaccID, p);

    if (tfaccID < 0 || tfaccID >= MAX_TFACC_CNT) {
        DPRINTK("TFACCID is not valid: %d\n", tfaccID);
        return -EBADMSG;
    }

    // DPRINTK("Try lock, pid: %d, tgid: %d, for tfacc: %d \n", current->pid, current->tgid, tfaccID);
    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfacc: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);

    // 如果已经持有锁，直接返回成功
    if (dev->holdTFACCPid[tfaccID] == current->pid) {
        DPRINTK("ALREADY HOLD TFACC, %d, tfacc: %d\n", current->pid, tfaccID);
        return 0;
    }

    while (true) {
        // 尝试进入临界区
        // DPRINTK("[ENTERING ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        mutex_lock(&app_mutex);
        // DPRINTK("[ENTERED ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        if (dev->holdTFACCPid[tfaccID] == -1) {
            dev->holdTFACCPid[tfaccID] = current->pid;
            dev->holdTFACCTgid[tfaccID] = current->tgid;
            lock_pid = current->pid;
            insertAppLockRecord(dev, tfaccID);
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);

        // 检查是否拿到TFACC
        if (lock_pid == current->pid) {
            // DPRINTK("GOT TFACC SUCC: %d, tfacc: %d\n", current->pid, tfaccID);
            return 0;
        }

        // 如果只是尝试一次，那么失败立即返回
        if (sleepUs < 0) {
            DPRINTK("FAIL to get tfacc once: %d\n", current->pid);
            return -EBADMSG;
        }

        // 如果 = 0 ，表示不断重试
        if (sleepUs == 0) {
            // udelay(10);
            DPRINTK("Not supported wait until: %d\n", current->pid);
            return -EBADMSG;
        } else { // 不断重试, 直到时间
            udelay(10);
            uWait -= 10;
            if (uWait < 0) {
                DPRINTK("FAIL to get tfacc for useconds: %d pid: %d, tgid: %d, tfacc: %d\n",
                        sleepUs, current->pid, current->tgid, tfaccID);
                return -EBADMSG;
            }
        }
    }
    return 0;
}

static int tf_app_try_unlock(struct tf_device* dev, int* p) {
    int tfaccID = -1;
    if (efuse) {
        return -1;
    }
    __get_user(tfaccID, (int*) (p) );

    if (tfaccID < 0 || tfaccID >= MAX_TFACC_CNT) {
        DPRINTK("TFACCID is not valid: %d\n", tfaccID);
        return -EBADMSG;
    }

    // DPRINTK("Try unlock, pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfaccID: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);

    {
        // 正常退出判断
        // DPRINTK("[ENTERING ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        mutex_lock(&app_mutex);
        // DPRINTK("[ENTERED ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        if (dev->holdTFACCPid[tfaccID] == current->pid) {
            finishAppLockRecord(dev, tfaccID);
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
    }

    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfaccID: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);
    return 0;
}

static int tf_app_release_tgid_lock(struct tf_device* dev) {
    int i;
    if (efuse) {
        return -1;
    }
    DPRINTK("Releasing, pid: %d, tgid: %d \n", current->pid, current->tgid);
    // DPRINTK("Current lock holder pid: %d, tgid: %d \n", dev->holdAppLockPid, dev->holdAppLockTgid);

    // 意味着进程组退出, 需要退出pid, 以及检查当前持有锁的tgid
    {
        // DPRINTK("TG [ENTERING ZONE], pid: %d, tgid: %d\n", current->pid, current->tgid);
        mutex_lock(&app_mutex);
        // DPRINTK("TG [ENTERED ZONE], pid: %d, tgid: %d\n", current->pid, current->tgid);

        for (i = 0; i < MAX_TFACC_CNT; ++i) {
            if (dev->holdTFACCPid[i] == current->pid) {
                finishAppLockRecord(dev, i);
            } else {
                if (dev->holdTFACCTgid[i] == current->tgid) {
                    finishAppLockRecord(dev, i);
                }
            }
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d. \n", current->pid, current->tgid);
        // DPRINTK("TG [QUIT LOCK], pid: %d, tgid: %d\n", current->pid, current->tgid);
    }

    for (i = 0; i < MAX_TFACC_CNT; ++i) {
        DPRINTK("Current lock holder for tfacc: %d, pid: %d, tgid: %d \n",
                i, dev->holdTFACCPid[i], dev->holdTFACCTgid[i]);
    }
    // DPRINTK("Current lock holder pid: %d, tgid: %d \n", dev->holdAppLockPid, dev->holdAppLockTgid);
    return 0;
}

/// 获得所有锁获取记录
static int tf_get_app_lock_records(struct tf_lock_record* p) {
    int i;
    int ret;
    if (efuse) {
        return -1;
    }
    for (i = 0; i < MAX_APP_LOCK_RECORDS; ++i) {
        if (app_lock_records[i].isHolding) app_lock_records[i].unlockTime = jiffies;
        ret = copy_to_user((void *) p, &app_lock_records[i], sizeof(struct tf_lock_record));
        p++;
    }
    return 0;
}

// 获取使用率状态
// p[0]: 15s之内的总使用率
// p[1]: 60s之内的总使用率
// p[2]: 300s之内的总使用率
// p[3 ~ 34]: p[i]代表(i - 3)号tfacc在15s内的使用率，如果为-1代表这个tfacc不存在，否则为0~1000之间的数代表利用率 * 1000
static int tf_get_app_usage(int *pp) {
    int i, j;
    int p[MAX_TFACC_CNT + 5];
    int s1, s2;
    struct tf_use_record *record;
    unsigned int curMs;
    for (i = 0; i < MAX_TFACC_CNT + 3; i++) {
        __put_user(-1, pp + i);
        p[i] = -1;
    }
    if (efuse) {
        return -1;
    }
    p[0] = p[1] = p[2] = 0;

    for (i = 0; i < chips * 8; i++) {
        p[3 + i] = 0;
        s1 = 0;
        s2 = 0;
        curMs = jiffies_to_msecs(jiffies);
        updateUseRecord(i, useRecords[i].isHolding, curMs);
        record = &useRecords[i];
        for (j = 0; j < record->len; j++) {
            int pos = (record->head + j) % RECORD_QUEUE_LEN;
            int gap = (curMs - record->startTime[pos]) / 1000 + 1;
            if (gap <= 15) {
                p[3 + i] += record->percent[pos];
            }
            if (gap <= 60) {
                s1 += record->percent[pos];
            }
            if (gap <= 300) {
                s2 += record->percent[pos];
            }
        }
        p[3 + i] += record->lastUseTime;
        p[3 + i] *= 1000;
        p[3 + i] /= (14000 + (record->lastRecordTime - record->lastTimeInQueue));
        p[0] += p[3 + i];

        s1 *= 1000;
        s1 /= (59000 + (record->lastRecordTime - record->lastTimeInQueue));

        s2 *= 1000;
        s2 /= (299000 + (record->lastRecordTime - record->lastTimeInQueue));

        p[1] += s1;
        p[2] += s2;
    }

    for (i = 0; i < MAX_TFACC_CNT + 3; i++) {
        __put_user(p[i], pp + i);
    }
    return 0;
}

/* pids using this driver, TODO use a struct to collect process info */
struct mutex pid_mutex;
#define MAXPINLIST 64
struct tf_app_info app_infos[MAXPINLIST];

void push_pid(void) {
    int pid = current->pid;
    int i = 0;

    if (efuse) {
        return;
    }
    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == pid) {
            mutex_unlock(&pid_mutex);
            return;
        }
    }
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == -1) {
            app_infos[i].pid = pid;
            app_infos[i].tgid = current->tgid;
            mutex_unlock(&pid_mutex);
            return;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void pop_pid(void) {
    int i = 0;
    int pid = current->pid;
    if (efuse) {
        return;
    }

    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == pid) {
            app_infos[i].pid = -1;
            app_infos[i].tgid = -1;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void pop_tgid(void) {
    int i = 0;
    if (efuse) {
        return;
    }

    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].tgid == current->tgid) {
            app_infos[i].pid = -1;
            app_infos[i].tgid = -1;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void get_pids(int* p) {
    int offset = 0;
    if (efuse) {
        return;
    }

    mutex_lock(&pid_mutex);
    for (offset = 0; offset < MAXPINLIST; ++offset) {
        if (app_infos[offset].pid >= 0) {
            __put_user(app_infos[offset].pid, p);
            p++;
        }
    }
    __put_user(-1, p);

    mutex_unlock(&pid_mutex);
    return;
}

static void init_pids(void) {
    // init pids
    int i = 0;

    if (efuse) {
        return;
    }

    for (i = 0; i < MAXPINLIST; ++i) {
        app_infos[i].pid = -1;
        app_infos[i].tgid = -1;
    }
    return;
}

static struct tf_device * tf_create_and_init_device(int device_id_counter) {
    int retval;
    struct tf_device * dev;
    int c;

    if (efuse) {
        return dev;
    }

    DPRINTK("ENTER\n");

    /* dev zeroed in alloc_etherdev */
    if (!(dev = (struct tf_device *) kzalloc(sizeof(struct tf_device), GFP_KERNEL))) {
        DPRINTK("failed to alloc tf_device\n");
        retval = -ENOMEM;
        goto fail_alloc_device;
    }

    spin_lock_init(&dev->lock);
    dev->isBusy = 0;
    // dev->holdAppLockPid = -1;
    for (c = 0; c < MAX_TFACC_CNT; ++c) {
        dev->holdTFACCPid[c] = -1;
        dev->holdTFACCTgid[c] = -1;
    }
    DPRINTK("dev tfacc inited\n");

    DPRINTK("isBusy = %d\n", dev->isBusy);
    tf_dev = dev;
    dev->mmap_id_counter = 1;

    for (c = 0; c < chips; c++) {
        int i;
        unsigned long long gap = chipGap * c;
        unsigned int TFACC_BASE[TFACC_REG_CNT] = {TFACC0_BASE, TFACC1_BASE, TFACC2_BASE, TFACC3_BASE,
                                                  TFACCLITE0_BASE, TFACCLITE1_BASE, TFACCLITE2_BASE, TFACCLITE3_BASE};
        unsigned int TFACC_CACHE_BASE[TFACC_REG_CNT] = {
                TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE,
                TFACCLITE0_CACHE_BASE, TFACCLITE1_CACHE_BASE, TFACCLITE2_CACHE_BASE, TFACCLITE3_CACHE_BASE
        };
        for (i = 0; i < TFACC_REG_CNT; i++) {
            unsigned long long base = gap + TFACC_BASE[i];
            int index = i + c * TFACC_REG_CNT;
            dev->ioreg[index] = ioremap(base, DEVICE_IO_LENGTH);
            if (dev->ioreg[index] == NULL || (unsigned long long)dev->ioreg[index] == 0xFFFFFFFF) {
                DPRINTK("failed to map io reg!\n");
                goto fail_remap;
            }
            DPRINTK("version: 0x%x\n", *(volatile unsigned int *)(dev->ioreg[index]));
            dev->reg_buf[index].phy_addr = base;
            dev->reg_buf[index].kernel_addr = dev->ioreg[index];
            dev->reg_buf[index].len = DEVICE_IO_LENGTH;
            dev->reg_buf[index].mmap_id = (index == 0 ? 0 : REG2ID + (index - 1));

            base = gap + TFACC_CACHE_BASE[i];
            dev->ioreg_cache[index] = ioremap(base, DEVICE_IO_LENGTH);
            if (dev->ioreg_cache[index] == NULL || (unsigned long long)dev->ioreg_cache[index] == 0xFFFFFFFF) {
                DPRINTK("failed to map io reg_cache!\n");
                goto fail_remap;
            }
            dev->cache_reg_buf[index].phy_addr = base;
            dev->cache_reg_buf[index].kernel_addr = dev->ioreg_cache[index];
            dev->cache_reg_buf[index].len = DEVICE_IO_LENGTH;
            dev->cache_reg_buf[index].mmap_id = CACHEREGID + index;
        }
    }
/*
    dev->cbuf.phy_addr = 0xF4400000;
    dev->cbuf.kernel_addr = ioremap(0xF4400000, 0x200000);
    dev->cbuf.len = 0x200000;
    dev->cbuf.mmap_id = CBUF;
*/
    dev->cbuf.phy_addr = reserveDDRBlocks[15].startPos;
    dev->cbuf.len = 256 * 1024 * 1024;
    dev->cbuf.kernel_addr = ioremap(dev->cbuf.phy_addr, dev->cbuf.len);
    dev->cbuf.mmap_id = CBUF;

    DPRINTK("EXIT, succeed\n");
    return dev;

    fail_remap:
    kfree(dev);
    dev = NULL;

    fail_alloc_device:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return NULL;
}

static void tf_remove_device_buf(struct tf_device * dev) {
    if (efuse) {
        return;
    }

    DPRINTK("ENTER\n");
    DPRINTK("EXIT, succeed\n");
}

static int tf_create_and_init_cdev(struct tf_device * dev, int device_id) {
    dev_t chrdev;
    int retval;

    if (efuse) {
        return -1;
    }

    assert(dev != NULL);

    // register a char device
    dev->minor = device_id;
    if ((retval = alloc_chrdev_region(&chrdev, dev->minor, 1, "thinkforce")) < 0) {
        DPRINTK("failed to alloc chrdev\n");
        goto origin;
    }

    dev->major = MAJOR(chrdev);
    DPRINTK("Major: %d, Minor:%d\n", dev->major, dev->minor);
    dev->device = device_create(thinkforce_class, NULL, chrdev, NULL, "thinkforce" "%d", dev->minor);
    if (!dev->device) {
        DPRINTK("failed to create cdev\n");
        goto after_alloc;
    }

    cdev_init(&dev->cdev, &tf_device_ops);
    dev->cdev.owner = THIS_MODULE;
    if ((retval = cdev_add(&dev->cdev, chrdev, 1))) {
        DPRINTK("failed to add cdev %d\n", device_id);
        goto after_create;
    }
    dev_set_drvdata(dev->device, dev);

/*
    DPRINTK("dev->device->dma_mem: 0x%p\n", dev->device->dma_mem);
*/
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_create:
    device_destroy(thinkforce_class, chrdev);

    after_alloc:
    unregister_chrdev_region(chrdev, 1);

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);

    return retval;
}

static void tf_remove_cdev(struct tf_device *dev) {
    dev_t chrdev;
    if (efuse) {
        return;
    }
#if 0
    volatile unsigned int *cache0_apb = ioremap(TFACC0_CACHE_BASE, DEVICE_IO_LENGTH);
    int i;
    for (i = 0; i < 64; i++) {
        DPRINTK("cache0_apb[0x%x] = 0x%x\n", 0x100 + i * 4, cache0_apb[0x40 + i]);
    }
    iounmap(cache0_apb);

    volatile unsigned int *cache1_apb = ioremap(TFACC1_CACHE_BASE, DEVICE_IO_LENGTH);
    for (i = 0; i < 64; i++) {
        DPRINTK("cache1_apb[0x%x] = 0x%x\n", 0x100 + i * 4, cache1_apb[0x40 + i]);
    }
    iounmap(cache1_apb);

    volatile unsigned int *reg = ioremap(TFACC0_CACHE_BASE, DEVICE_IO_LENGTH);
    for (i = 0; i < 128; i++) {
        DPRINTK("reg[0x%x] = 0x%x\n", i * 4, reg[i]);
    }

    DPRINTK("reg[0x004] = 0x%x\n", reg[0x004 / 4]);
    DPRINTK("reg[0x198] = 0x%x\n", reg[0x198 / 4]);
    DPRINTK("reg[0x19C] = 0x%x\n", reg[0x19C / 4]);
    DPRINTK("reg[0x1A0] = 0x%x\n", reg[0x1A0 / 4]);
    DPRINTK("reg[0x1A4] = 0x%x\n", reg[0x1A4 / 4]);
    DPRINTK("reg[0x1A8] = 0x%x\n", reg[0x1A8 / 4]);
    DPRINTK("reg[0x1AC] = 0x%x\n", reg[0x1AC / 4]);
    DPRINTK("reg[0x1B0] = 0x%x\n", reg[0x1B0 / 4]);
    DPRINTK("reg[0x1B4] = 0x%x\n", reg[0x1B4 / 4]);
    DPRINTK("reg[0x1B8] = 0x%x\n", reg[0x1B8 / 4]);
    DPRINTK("reg[0x1BC] = 0x%x\n", reg[0x1BC / 4]);
    DPRINTK("reg[0x1C0] = 0x%x\n", reg[0x1C0 / 4]);
    DPRINTK("reg[0x1C4] = 0x%x\n", reg[0x1C4 / 4]);
#endif
    DPRINTK("ENTER\n");

    //remove cdev
    chrdev = MKDEV(dev->major, dev->minor);
    cdev_del(&dev->cdev);
    device_destroy(thinkforce_class, chrdev);
    unregister_chrdev_region(chrdev, 1);

    DPRINTK("EXIT, succeed\n");
}

void tfacc_full_swrst(unsigned long long, unsigned long long);
void tfacc_lite_swrst(unsigned long long);
void tfacc_enable_one_cache(unsigned long long);

static int tf_release(struct inode *inode, struct file *filp) {
    struct tf_device *dev = container_of(inode->i_cdev, struct tf_device, cdev);
    int i;
    unsigned long long gap;

    if (efuse) {
        return -1;
    }

    DPRINTK("release ENTER\n");
    spin_lock(&dev->lock);
    dev->isBusy--;
    DPRINTK("isBusy = %d\n", dev->isBusy);

    for (i = 0; i < reserveDDRBlcokCnt; i++) {
        if (reserveDDRBlocks[i].tgid == current->tgid) {
            reserveDDRBlocks[i].isMalloc = false;
            reserveDDRBlocks[i].offset = 0;
        }
    }

    if (dev->isBusy == 0) {
        for (i = 0; i < reserveDDRBlcokCnt; i++) {
            reserveDDRBlocks[i].isMalloc = false;
            reserveDDRBlocks[i].offset = 0;
        }

        tf_remove_device_buf(dev);
        dev->mmap_id_counter = 1;
    }

    // 进程退出的时候，把整个线程组的锁都释放掉
    // if (current->pid == current->tgid) {
    tf_app_release_tgid_lock(dev);
    pop_tgid();
    // } else {
    // tf_app_try_unlock(dev, NULL);
    // pop_pid();
    // }
    if (dev->isBusy == 0)
    {
        int c;
        for (c = chips-1; c>=0; c--) {
            gap = chipGap * c;
            //tfacc_full_swrst(gap + TFACC0_BASE, gap + TFACC1_BASE);
            //tfacc_full_swrst(gap + TFACC2_BASE, gap + TFACC3_BASE);
            tfacc_lite_swrst(gap + TFACCLITE0_BASE);
            tfacc_lite_swrst(gap + TFACCLITE1_BASE);
            tfacc_lite_swrst(gap + TFACCLITE2_BASE);
            tfacc_lite_swrst(gap + TFACCLITE3_BASE);

            tfacc_enable_one_cache(TFACC0_CACHE_BASE);
            tfacc_enable_one_cache(TFACC1_CACHE_BASE);
            tfacc_enable_one_cache(TFACC2_CACHE_BASE);
            tfacc_enable_one_cache(TFACC3_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE0_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE1_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE2_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE3_CACHE_BASE);

        }
    }

    spin_unlock(&dev->lock);

    DPRINTK("EXIT, succeed\n");
    return 0;
}

static int tf_mmap(struct file *filp, struct vm_area_struct *vma) {
    struct tf_device *dev = (struct tf_device *)filp->private_data;
    int size = vma->vm_end - vma->vm_start;
    int mmap_id = vma->vm_pgoff;
    dma_addr_t phy_addr = -1;
    int max_size = 0;
    struct kbuf * kbuf_p;
    void *kernel_addr;

    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER\n");
    DPRINTK("mapping typeid: %d\n", mmap_id);
/*
    if (!versionIsRight) {
        DPRINTK("SDK's version is wrong.\n");
        return -EINVAL;
    }
*/
    if (mmap_id == SRAM1ID) {
        //sram1
        DPRINTK("prepare mapping sram1\n");
        max_size = dev->sram1_buf.len;
        phy_addr = dev->sram1_buf.phy_addr;
        kernel_addr = dev->sram1_buf.kernel_addr;
    } else if (mmap_id == SRAM2ID) {
        //sram2
        DPRINTK("prepare mapping sram2\n");
        max_size = dev->sram2_buf.len;
        phy_addr = dev->sram2_buf.phy_addr;
        kernel_addr = dev->sram2_buf.kernel_addr;
    } else if (mmap_id == 0) {
        //reg
        DPRINTK("prepare mapping reg\n");
        max_size = dev->reg_buf[0].len;
        phy_addr = dev->reg_buf[0].phy_addr;
        kernel_addr = dev->reg_buf[0].kernel_addr;
    } else if (mmap_id >= REG2ID && mmap_id < REG2ID + TFACC_REG_CNT * chips - 1) {
        DPRINTK("prepare mapping reg2\n");
        max_size = dev->reg_buf[mmap_id - REG2ID + 1].len;
        phy_addr = dev->reg_buf[mmap_id - REG2ID + 1].phy_addr;
        kernel_addr = dev->reg_buf[mmap_id - REG2ID + 1].kernel_addr;
    } else if (mmap_id >= CACHEREGID && mmap_id < CACHEREGID + TFACC_REG_CNT * chips) {
        DPRINTK("prepare mapping cache reg\n");
        max_size = dev->cache_reg_buf[mmap_id - CACHEREGID].len;
        phy_addr = dev->cache_reg_buf[mmap_id - CACHEREGID].phy_addr;
        kernel_addr = dev->cache_reg_buf[mmap_id - CACHEREGID].kernel_addr;
    } else if (mmap_id == REGMAINID) {
        DPRINTK("prepare mapping regMain\n");
        max_size = dev->regMain_buf.len;
        phy_addr = dev->regMain_buf.phy_addr;
        kernel_addr = dev->regMain_buf.kernel_addr;
    } else if (mmap_id < dev->mmap_id_counter) {
        DPRINTK("prepare mapping buf %d\n", mmap_id);
        hash_for_each_possible(dev->buf_list, kbuf_p, list, mmap_id & TF_BUF_HASHMASK) {
            if (kbuf_p->mmap_id == mmap_id) {
                max_size = kbuf_p->len;
                phy_addr = kbuf_p->phy_addr;
                kernel_addr = kbuf_p->kernel_addr;
                break;
            }
        }
    } else if (mmap_id == CBUF) {
        max_size = dev->cbuf.len;
        phy_addr = dev->cbuf.phy_addr;
        kernel_addr = dev->cbuf.kernel_addr;
    } else {
        DPRINTK("invalid offset\n");
        return -EINVAL;
    }

    DPRINTK("size: %d, max_size: %d\n", size, max_size);

    if (size > max_size) {
        DPRINTK("require mmap size too large\n");
        return -EINVAL;
    }

    DPRINTK("mmap phy_addr: 0x%llx, size: %d\n", phy_addr, size);
    vma->vm_flags |= VM_LOCKED;  //disable swap

    DPRINTK("PAGE_SHIFT: %d\n", PAGE_SHIFT);
    if (!mmap_id || (mmap_id >= REG2ID && mmap_id < REG2ID + TFACC_REG_CNT * chips - 1) ||
        (mmap_id >= CACHEREGID && mmap_id < CACHEREGID + TFACC_REG_CNT * chips)) {
        int r;
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        r = remap_pfn_range(vma, vma->vm_start, phy_addr >> PAGE_SHIFT, size, vma->vm_page_prot);
        if (r != 0) {
            DPRINTK("remap page range failed: %d\n", r);
            return -ENXIO;
        }
    } else {
        int r;
        vma->vm_pgoff = 0;
        //DPRINTK("kernel_addr: 0x%p, phy_addr: 0x%p\n", kernel_addr, (void *)phy_addr);
        if (isNextUncache == 1) {
            //vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
            isNextUncache = 0;
        }

        //vma->vm_page_prot = pgprot_cached(vma->vm_page_prot);
        r = remap_pfn_range(vma, vma->vm_start, phy_addr >> PAGE_SHIFT, size, vma->vm_page_prot);
        if (r < 0) {
            DPRINTK("mmap coherent failed: %d\n", r);
            return -ENXIO;
        }
    }

    DPRINTK("EXIT, succeed\n");
    return 0;
}

void tfacc_full_clap(unsigned long long base) {
    volatile unsigned int* clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    *(clk + 0x84) |= 0x3;
    iounmap(clk);
}
void tfacc_full_unclap(unsigned long long base) {
    volatile unsigned int* clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    *(clk + 0x84) &= ~0x3;
    iounmap(clk);
}

void tfacc_full_acp(unsigned long long base, int highAddr) {
    volatile unsigned int *buf = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    DPRINTK("base = 0x%llx, highAddr = 0x%x\n", base, highAddr);
    if (buf != NULL) {
        unsigned int temp = buf[0x10 / 4];
        DPRINTK("prot = 0x%x\n", temp);
        buf[0x10 / 4] = (buf[0x10 / 4] & (~(7 << 8))) | (2 << 8);
        buf[0x10 / 4] = (buf[0x10 / 4] & (~(7 << 12))) | (2 << 12);
        buf[0x10 / 4] = (buf[0x10 / 4] & (~(255 << 16))) | (highAddr << 16); // high addr
        buf[0x10 / 4] = (buf[0x10 / 4] & (~(255 << 24))) | (highAddr << 24); // high addr
        DPRINTK("prot = 0x%x\n", buf[0x10 / 4]);

        if (base == TFACC3_FULL_ACP_BASE) {
            buf[0xc4 / 4] |= (0x3 << 2);
            buf[0xc8 / 4] |= (0x7FFFF << 16);
        } else {
            buf[0x98 / 4] |= 3;
            buf[0x9c / 4] |= 0x7FFFF;
        }

        buf[0x90 / 4] |= (1 << 28) + (1 << 8) + (1 << 24) + (highAddr << 0) + (highAddr << 16); // (1 << 0) 和 (1 << 16)是高位地址
        buf[0x94 / 4] |= (1 << 28) + (1 << 8) + (1 << 24) + (highAddr << 0) + (highAddr << 16); // (1 << 0) 和 (1 << 16)是高位地址
        iounmap(buf);
    }
}

void tfacc_swrst_one_cache(unsigned long long base) {
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    //unsigned int cache_num=0;

    if (efuse) {
        return;
    }
    //volatile unsigned int *offset = apb + (cache_num * (1<<22) / sizeof(unsigned int));
    apb[0xe0 / 4] = 0x0;
    apb[0xf8 / 4] = 0x0;
    apb[0xfc / 4] = 0x0;

    iounmap(apb);
}
void tfacc_enable_one_cache(unsigned long long base) {
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    int data = 0;
    if (efuse) {
        return;
    }

    apb[0x50 / 4] = 1;
    __asm__ __volatile__ ("dmb sy");
    while (data != 1) {
        data = apb[0x54 / 4];
    }
    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 6);


    iounmap(apb);
}
void tfacc_enable_one_uncache(unsigned long long base) {
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }

    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 6);
    apb[0x10 / 4] = 0x100; // 31:0
    apb[0x14 / 4] = 0x0;   // 31:0
    apb[0x30 / 4] = 0x0;   // 63:32
    apb[0x34 / 4] = 0x0;   // 63:32
    apb[0x20 / 4] = 0x100; // 31:0
    apb[0x24 / 4] = 0x0;   // 31:0
    apb[0x40 / 4] = 0x0;   // 63:32
    apb[0x44 / 4] = 0x0;   // 63:32

    apb[0x18 / 4] = 0x0;          // 31:0
    apb[0x1C / 4] = 0xffffffff;   // 31:0
    apb[0x38 / 4] = 0x0;          // 63:32
    apb[0x3C / 4] = 0xffffffff;   // 63:32
    apb[0x28 / 4] = 0x0;          // 31:0
    apb[0x2C / 4] = 0xffffffff;   // 31:0
    apb[0x48 / 4] = 0x0;          // 63:32
    apb[0x4C / 4] = 0xffffffff;   // 63:32


    iounmap(apb);
}

void tfacc_full_enable_interleave(unsigned long long base) {
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }
    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 12);
    DPRINTK("interleave = 0x%x\n", apb[0x04 / 4]);
    iounmap(apb);
}

void tfacc_lite_swrst(unsigned long long BASE) {
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }
    reg[0xff] = 0xc012 | (0x1<<31) | (0x1<<5);
    iounmap(reg);
}
void tfacc_full_swrst(unsigned long long BASE, unsigned long long BASE1) {
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    unsigned int *reg1 = ioremap(BASE1, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }

    reg1[0xff] = (0x1<<5);
    reg1[0xfa]= (0x100);
    reg[0xfa] = (0x100);
    reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
    __asm__ __volatile__ ("dmb sy");


    iounmap(reg);
    iounmap(reg1);
}

void tfacc_lite_enable_cache(unsigned long long BASE, unsigned long long CACHE_BASE) {
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }

    if (!cacheInit) {
        tfacc_swrst_one_cache(CACHE_BASE);
        reg[0xfa] = 0x1;
        reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
        ++cacheInitCnt;
        if (cacheInitCnt == 6*chips)
            cacheInit = 1;
    }
    tfacc_enable_one_cache(CACHE_BASE);
    iounmap(reg);
}

void tfacc_full_enable(unsigned long long base);
void tfacc_full_disable(unsigned long long base);

void tfacc_full_enable_cache(unsigned long long BASE, unsigned long long BASE1, unsigned long long CACHE0_BASE, unsigned long long CACHE1_BASE,
                             unsigned long long CLKBASE, unsigned long long CFGBASE) {
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    unsigned int *reg1 = ioremap(BASE1, DEVICE_IO_LENGTH);
    unsigned int *cfgreg = ioremap(CFGBASE, TFACC_CLK_LENGTH);
    //unsigned int *clkreg = ioremap(CLKBASE, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }

#if 1
    if (!cacheInit) {

//        tfacc_full_clap(CLKBASE);
        tfacc_swrst_one_cache(CACHE1_BASE);
//        tfacc_full_unclap(CLKBASE);

//        reg[0xfa] |= 0x1;
//        reg[0xff] = 0xc012 + (0x1 << 31);
//        __asm__ __volatile__ ("dmb sy");
//        reg[0xfa] = 0;
//        reg[0xff] = 0xc012;

//        tfacc_full_clap(CLKBASE);
//        tfacc_full_unclap(CLKBASE);
        tfacc_swrst_one_cache(CACHE0_BASE);

        //cfgreg[0x28/4] = 0x0;
        //cfgreg[0x20/4] = 0x1;

        //Reset Mau and set reset with Cache
        reg1[0xfa]|= (0x1|0x100);
        reg[0xfa] |= (0x1|0x100);

        //Manually reset bufaptr
        reg1[0xff] = (0x1<<5);
        reg[0xff] = (0x1<<5);

        reg1[0xff] |= (0x1<<5);
        reg1[0xff] &= ~(0x1<<5);

        reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
        __asm__ __volatile__ ("dmb sy");

        reg[0xfa] = 0;
        reg1[0xfa] = 0;

        //if ((cfgreg[0x2c/4]&0x7)!=0) {
        //    DPRINTK("glitch at reset\n");
        //}
        //cfgreg[0x28/4] = 0xffffffff;
        //cfgreg[0x20/4] = 0x0;

//        tfacc_full_disable(CLKBASE);
//        tfacc_full_enable(CLKBASE);


        ++cacheInitCnt;
        if (cacheInitCnt == 6*chips)
            cacheInit = 1;

    }
    iounmap(reg);
    iounmap(reg1);
    iounmap(cfgreg);
    //iounmap(clkreg);
    tfacc_enable_one_cache(CACHE0_BASE);
    tfacc_enable_one_cache(CACHE1_BASE);
#else
    iounmap(reg);
    iounmap(reg1);
        ++cacheInitCnt;
        tfacc_enable_one_cache(CACHE0_BASE);
        tfacc_enable_one_cache(CACHE1_BASE);
#endif


    //tfacc_full_enable_interleave(CACHE0_BASE);
    //tfacc_full_enable_interleave(CACHE1_BASE);
}

void tfacc_full_enable(unsigned long long base) {
    unsigned int pllstatus;
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }

    if (clk != NULL) {
        *(clk + 0x24) = 0;          // unlock
        if (!chipInit) {
            DPRINTK("Reset tfacc%d pll\n", chipInitCnt);
            *(clk + 0x21) = 1;
            *(clk + 0x22) &= ~(1<<24);
            udelay(400);
            *(clk + 0x22) |= (1<<24);
            udelay(100);
            pllstatus = *(clk+0xf0);
            while ((pllstatus&0x3) != 0x3)
                pllstatus = *(clk+0xf0);
            chipInitCnt++;
            if (chipInitCnt == 2*chips)
                chipInit = 1;
        }
        *(clk + 0x86) |= 6;
        *(clk + 0x82) |= 2;
        *(clk + 0x84) &= ~0x3;
        iounmap(clk);
    }
}

void tfacc_full_checkrstcond(void) {
    unsigned int* periscfg = ioremap(TFACC0_FULL_ACP_BASE, TFACC_CLK_LENGTH);
    int retval;
    if (efuse) {
        return;
    }
    if (needReset == 1) periscfg[0x34/4] = 1;
    retval = periscfg[0x34/4];
    DPRINTK("Flag = %llx\n", (unsigned long long)retval);
    if (retval == 1) {
        periscfg[0x34/4] = 2;
        skipFullSwRst = 2;
    }
    iounmap(periscfg);
}

void tfacc_full_disable(unsigned long long base) {
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    DPRINTK("ioremap %llx\n", base);
    DPRINTK("ioremap %llx finish\n", base);

    if (clk != NULL) {
        *(clk + 0x24) = 0;          // unlock
        *(clk + 0x84) |= 0x3;
        if (skipFullSwRst != 2) {
            *(clk + 0x82) &= 0xFFFFFFFD;
            udelay(1);
            *(clk + 0x82) |= 2;
            udelay(10);
        } else {
            DPRINTK("Skip HW Reset\n");
        }
        *(clk + 0x86) &= 0xFFFFFFF9;

        //*(clk + 0x22) &= ~(1<<24);
        iounmap(clk);
    }
}

void tfacc_lite_enable(unsigned long long base) {
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    if (clk != NULL) {
        *(clk + 0x24) = 0;
        *(clk + 0x86) |= 0x3C0000;
        *(clk + 0x82) |= 0x600;
        *(clk + 0x84) &= ~(0x3<<9);
        iounmap(clk);
    }
}

void tfacc_lite_disable(unsigned long long base) {
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (efuse) {
        return;
    }
    if (clk != NULL) {
        *(clk + 0x24) = 0;
        *(clk + 0x84) |= (0x3<<9);
        if (skipFullSwRst !=2) {
            *(clk + 0x82) &= 0xFFFFF9FF;
        } else {
            DPRINTK("Skip HW Reset\n");
        }
        *(clk + 0x86) &= 0xFFC3FFFF;
        iounmap(clk);
    }
}
static void tf_ioctl_clear(struct tf_device * dev) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    tf_remove_device_buf(dev);
    dev->mmap_id_counter = 1;

    DPRINTK("EXIT, succeed\n");
}

static void tf_ioctl_reset(struct tf_device * dev) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    hardware_tfacc_reset();

    tfacc_full_enable(TFACC_BL_CLK_BASE);
    tfacc_full_enable(TFACC_BR_CLK_BASE);
    tfacc_lite_enable(TFACC_L_CLK_BASE);
    tfacc_lite_enable(TFACC_R_CLK_BASE);

    tfacc_full_acp(TFACC0_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC1_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC2_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC3_FULL_ACP_BASE, ddrStart >> 32);

    tfacc_full_enable_cache(TFACC0_BASE, TFACC1_BASE, TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC_BL_CLK_BASE, TFACC0_FULL_ACP_BASE);
    tfacc_full_enable_cache(TFACC2_BASE, TFACC3_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE, TFACC_BR_CLK_BASE, TFACC1_FULL_ACP_BASE);

    tfacc_lite_enable_cache(TFACCLITE0_BASE, TFACCLITE0_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE1_BASE, TFACCLITE1_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE2_BASE, TFACCLITE2_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE3_BASE, TFACCLITE3_CACHE_BASE);

    DPRINTK("EXIT, succeed\n");
}


static int tf_ioctl_check_version(struct tf_device * dev, void * arg) {
    struct tf_version *io_param;
    int retval;
    int minVersion = 1840;
    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER\n");
    if (!(io_param = (struct tf_version*) kmalloc(sizeof(struct tf_version), GFP_KERNEL))) {
        DPRINTK("fail to alloc io_param\n");
        retval = -ENOMEM;
        goto origin;
    }
    if ((retval = copy_from_user(io_param, (void *) arg, sizeof(struct tf_version)))) {
        DPRINTK("fail to copy io_param from user\n");
        goto after_alloc_io_param;
    }

    //check version
    if (io_param->sdk_version >= minVersion) {
        versionIsRight = true;
        io_param->kernel_version = 20190605;
    } else {
        io_param->kernel_version = -1;
        io_param->excepted_sdk_version = minVersion;
    }

    if ((retval = copy_to_user((void *) arg, io_param, sizeof(struct tf_version)))) {
        DPRINTK("fail to copy io_param to user\n");
        goto after_alloc_io_param;
    }

    kfree(io_param);
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_alloc_io_param:
    kfree(io_param);

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return retval;
}

static int tf_init_kbuf(struct tf_device * dev,struct kbuf * kbuf_p) {
    bool mallocOk;
    int i;
    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER\n");

    dma_set_mask(dev->device, DMA_BIT_MASK(32));
    dma_set_coherent_mask(dev->device, DMA_BIT_MASK(32));

    mallocOk = false;

    for (i = 0; i < reserveDDRBlcokCnt; i++) {
        if (reserveDDRBlocks[i].chipId == dev->useDDR2 &&
            reserveDDRBlocks[i].offset + kbuf_p->len <= reserveDDRBlocks[i].len) {
            if (!reserveDDRBlocks[i].isMalloc || reserveDDRBlocks[i].tgid == current->tgid) {
                kbuf_p->phy_addr = reserveDDRBlocks[i].startPos + reserveDDRBlocks[i].offset;
                reserveDDRBlocks[i].offset += kbuf_p->len;
                reserveDDRBlocks[i].isMalloc = true;
                reserveDDRBlocks[i].tgid = current->tgid;

                kbuf_p->mmap_id = dev->mmap_id_counter++;
                if (dev->mmap_id_counter == 9000) {
                    //中间这一段编号保留，另做他用
                    dev->mmap_id_counter = 11000;
                }

                mallocOk = true;
                break;
            }
        }
    }

    if (!mallocOk) {
        kbuf_p->phy_addr = 0x0;
        kbuf_p->mmap_id = -1;
    }

    DPRINTK("EXIT, succeed\n");
    return 0;
}

static struct kbuf * tf_create_and_init_kbuf(struct tf_device * dev, int len) {
    int retval;
    struct kbuf * kbuf_p;
    if (efuse) {
        return (struct kbuf *)-1;
    }

    DPRINTK("ENTER, succeed\n");
    if (!(kbuf_p = kzalloc(sizeof(struct kbuf), GFP_KERNEL))) {
        retval = -ENOMEM;
        goto origin;
    }
    kbuf_p->len = len;
    DPRINTK("kbuf_p->len: 0x%08x\n", len);
    if ((retval = tf_init_kbuf(dev, kbuf_p))) {
        DPRINTK("failed to init kbuf\n");
        goto after_alloc_kbuf;
    }
    DPRINTK("buf phy_addr: %llx, kernel_addr: %p, len: 0x%08x, mmap_id: 0x%08x\n",
            kbuf_p->phy_addr, kbuf_p->kernel_addr, kbuf_p->len, kbuf_p->mmap_id);
    DPRINTK("EXIT, succeed\n");
    return kbuf_p;

    after_alloc_kbuf:
    kfree(kbuf_p);
    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return NULL;
}


static int tf_ioctl_create(struct tf_device * dev, void * arg) {
    struct tf_buf_io_param *io_param;
    struct kbuf * kbuf_p;
    int retval;
    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER\n");
    if (!(io_param = (struct tf_buf_io_param*) kmalloc(sizeof(struct tf_buf_io_param), GFP_KERNEL))) {
        DPRINTK("fail to alloc io_param\n");
        retval = -ENOMEM;
        goto origin;
    }
    if ((retval = copy_from_user(io_param, (void *) arg, sizeof(struct tf_buf_io_param)))) {
        DPRINTK("fail to copy io_param from user\n");
        goto after_alloc_io_param;
    }

    dev->useDDR2 = io_param->useDDR2;

    //create kbuf
    if (!(kbuf_p = tf_create_and_init_kbuf(dev, io_param->len))) {
        DPRINTK("fail to create_and_init kbuf\n");
        retval = -ENOMEM;
        goto after_alloc_io_param;
    }

    if (io_param->uncache) {
        isNextUncache = 1;
    }

    //add to hlist
    hash_add(dev->buf_list, &kbuf_p->list,
             kbuf_p->mmap_id & TF_BUF_HASHMASK);
    //copy to user
    io_param->phy_addr = kbuf_p->phy_addr;
    io_param->mmap_id = kbuf_p->mmap_id;
    if ((retval = copy_to_user((void *) arg, io_param, sizeof(struct tf_buf_io_param)))) {
        DPRINTK("fail to copy io_param to user\n");
        goto after_add_to_hlist;
    }

    kfree(io_param);
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_add_to_hlist:
    hash_del(&kbuf_p->list);
#ifdef ZYNQMP
#elif defined(USE_DMA)
    dma_free_coherent(dev->device, kbuf_p->len, kbuf_p->kernel_addr, kbuf_p->phy_addr);
#else
    free_pages((unsigned long)kbuf_p->kernel_addr, get_order(kbuf_p->len));
#endif
    kfree(kbuf_p);
    after_alloc_io_param:
    kfree(io_param);
    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return retval;
}

static int tf_ioctl_get_app_infos(struct tf_device* dev, int* p) {
    DPRINTK("GET PID LIST\n");

    get_pids(p);
    return 0;
}

static long tf_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    struct tf_device *dev = (struct tf_device *) filp->private_data;
    long retval;
    if (efuse) {
        return -1;
    }

    // DPRINTK("ENTER: %u, type: %d NR: %d, %u \n", cmd, _IOC_TYPE(cmd), _IOC_NR(cmd), TF_READ_PIDS);
    if (_IOC_TYPE(cmd) != TF_MAGIC) return -EINVAL;
    if (_IOC_NR(cmd) >= TF_MAX_NR) return -EINVAL;
    switch (cmd) {
        case TF_VERSION_CHECK:
            retval = tf_ioctl_check_version(dev, (void*) arg);
            break;
        case TF_BUF_RESET:
            tf_ioctl_reset(dev);
            retval = 0;
            break;
        case TF_BUF_CLEAR:
            tf_ioctl_clear(dev);
            retval = 0;
            break;
        case TF_BUF_CREATE:
            retval = tf_ioctl_create(dev, (void *) arg);
            break;
        case TF_READ_PIDS:
            retval = tf_ioctl_get_app_infos(dev, (int*) arg);
            break;

        case TF_APP_LOCK:
            retval = tf_app_try_lock(dev, (int*) arg);
            break;

        case TF_APP_UNLOCK:
            retval = tf_app_try_unlock(dev, (int*) arg);
            break;

        case TF_READ_APP_LOCK_RECORD:
            retval = tf_get_app_lock_records((struct tf_lock_record*) arg);
            break;

        case TF_READ_APP_USAGE:
            retval = tf_get_app_usage((int*)arg);
            break;

        case TF_READ_RESERVE_MEM_RECORD:
            retval = tf_get_reserve_ddr_blocks((struct ReserveDDRBlock*) arg);
            break;
        default:
            DPRINTK("error cmd\n");
            retval = -EINVAL;
            break;
    }

    if (!retval) {
        // DPRINTK("EXIT, succeed\n");
        return 0;
    }
    DPRINTK("EXIT, failed with code %ld\n", retval);
    DPRINTK("FAIL CMD: %u, type: %d NR: %d \n", cmd, _IOC_TYPE(cmd), _IOC_NR(cmd));
    return retval;
}

void tf_remove(void) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    tf_remove_cdev(tf_dev);
    tf_remove_device_buf(tf_dev);
    dev_set_drvdata(tf_dev->device, NULL);
    kfree(tf_dev);

    DPRINTK("EXIT, succeed\n");
}

//reserve reserveSize M bytes.
//normal: 256
//1MFace: 512
//5MFace: 1536
//10MFace: 2560
long long reserveSize = 0;

static ssize_t show_kernel_version(struct device *dev,
                                   struct device_attribute *attr, char *buf)
{
    int ret;
    if (efuse) {
        return -1;
    }
    if (reserveSize == 256) {
        ret = sprintf(buf, "Normal\n");
    } else if (reserveSize == 512) {
        ret = sprintf(buf, "1M Face\n");
    } else if (reserveSize == 800) {
        ret = sprintf(buf, "3M Face\n");
    } else if (reserveSize == 1536) {
        ret = sprintf(buf, "5M Face\n");
    } else if (reserveSize == 2560) {
        ret = sprintf(buf, "10M Face\n");
    } else {
        ret = sprintf(buf, "Unknown\n");
    }

    return ret;
}

static ssize_t set_my_kernel(struct device *dev,
                             struct device_attribute *attr,
                             const char *buf, size_t len)
{
    if (efuse) {
        return -1;
    }
    return len;
}

static DEVICE_ATTR(kernel_version, S_IWUSR|S_IRUSR, show_kernel_version, set_my_kernel);


struct file_operations mytest_ops={
        .owner  = THIS_MODULE,
};

static int major;
//static struct class *cls;
//static struct class *profileCls;

void output_tfacc_id(unsigned long long base) {
    unsigned int *reg = ioremap(base, DEVICE_IO_LENGTH);
    if (efuse) {
        return;
    }
    if (reg) {
        printk("version: 0x%08X\n", *reg);
        printk("ID: 0x%08X\n", *(reg + 1));
        iounmap(reg);
    }
}

static void readSocketInfo(void) {
    unsigned int isDualSocket, isDualDie;
    unsigned int *configBase = ioremap(0xFE170000, 0x100000);
    long long perChip, perBlock;
    int c, i;
    unsigned long long gap;
    unsigned int *efuseAddr = ioremap(EFUSE_BASE, 0x100);

    if (efuseAddr[0] & 2) {
        efuse = 1;
    }

    // 读取chip信息
    isDualSocket = (*(volatile unsigned int *)(configBase + 0x30 / 4) & 0x40) >> 6;
    isDualDie = *(volatile unsigned int *)(configBase + 0x3046C / 4) & 0x1;
    iounmap(configBase);

    if (isDualSocket) chips = 4;
    else if (isDualDie) chips = 2;
    else chips = 1;

    if (chips == 2) {
        chipGap = 0x8000000000LL;
    } else if (chips == 4) {
        chipGap = 0x4000000000LL;
    }

    reserveDDRBlcokCnt = 0;
    perChip = 0x100000000;
    perBlock = 256 * 1024 * 1024;

    for (c = 0; c < chips; c++) {
        for (i = 0; i < perChip / perBlock; i++) {
            gap = chipGap * c;
            reserveDDRBlocks[reserveDDRBlcokCnt].chipId = c;
            reserveDDRBlocks[reserveDDRBlcokCnt].len = perBlock;
            reserveDDRBlocks[reserveDDRBlcokCnt].isMalloc = false;
            reserveDDRBlocks[reserveDDRBlcokCnt].offset = 0;
            reserveDDRBlocks[reserveDDRBlcokCnt].startPos = ddrStart + gap + perBlock * i;
            reserveDDRBlocks[reserveDDRBlcokCnt].tgid = -1;

            if (reserveDDRBlocks[reserveDDRBlcokCnt].startPos % 0x100000000 == 0) {
                reserveDDRBlocks[reserveDDRBlcokCnt].startPos += 1 * 1024 * 1024;
                reserveDDRBlocks[reserveDDRBlcokCnt].len -= 1 * 1024 * 1024;
            }
            reserveDDRBlcokCnt++;
        }
    }

    return;
}

static int tf_open(struct inode *inode, struct file *filp) {
    struct tf_device *dev = container_of(inode->i_cdev, struct tf_device, cdev);
    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER: pid: %d, tgid: %d\n", current->pid, current->tgid);
    spin_lock(&dev->lock);
    DPRINTK("isBusy = %d\n", dev->isBusy);
    filp->private_data = dev;
    dev->isBusy++;
#if 0
    if (dev->isBusy == 1) {
        //if (chips == 1) {
        //    //hardware_tfacc_reset();
        //}

        tfacc_full_enable(TFACC_BL_CLK_BASE);
        tfacc_full_enable(TFACC_BR_CLK_BASE);
        tfacc_lite_enable(TFACC_L_CLK_BASE);
        tfacc_lite_enable(TFACC_R_CLK_BASE);

        tfacc_full_acp(TFACC0_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC1_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC2_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC3_FULL_ACP_BASE, ddrStart >> 32);

        tfacc_full_enable_cache(TFACC0_BASE, TFACC1_BASE, TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC_BL_CLK_BASE);
        tfacc_full_enable_cache(TFACC2_BASE, TFACC1_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE, TFACC_BR_CLK_BASE);

        tfacc_lite_enable_cache(TFACCLITE0_BASE, TFACCLITE0_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE1_BASE, TFACCLITE1_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE2_BASE, TFACCLITE2_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE3_BASE, TFACCLITE3_CACHE_BASE);
    }
#endif

    push_pid();
    DPRINTK("current pid: %d\n", current->pid);
    spin_unlock(&dev->lock);

    DPRINTK("EXIT, succeed\n");
    return 0;
}

#ifdef CONFIG_ACPI
static int __init tf_init_module(struct platform_device *pdev)
#else
static int __init tf_init_module(void)
#endif
{
#ifdef CONFIG_ACPI
    struct resource *res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
#endif
    int retval;
    struct tf_device *dev;
    int c;
    unsigned long long gap;
    if (efuse) {
        return -1;
    }

    //struct device *mydev;
    DPRINTK("page = %d\n", PAGE_SHIFT);
    DPRINTK("ENTER\n");

#ifdef CONFIG_ACPI
    ddrStart = res->start;
#else
    // 读取reserveDDR的信息
    struct device_node *reserved0 = of_find_node_by_path("/reserved-memory/buffer@1");
    uint8_t *r0 = (uint8_t*)reserved0->properties[1].value;
    ddrStart = (((long long)r0[3] << 32) + ((long long)r0[4] << 24) + ((long long)r0[5] << 16) + ((long long)r0[6] << 8) + (long long)r0[7]);
#endif

    DPRINTK("ddrStart: 0x%llx\n", (unsigned long long)ddrStart);

    readSocketInfo();
    DPRINTK("EFUSE = %d\n", efuse);
    DPRINTK("chips = %d\n", chips);
    if (efuse) {
        return -1;
    }

    major = register_chrdev(0, "thinkforce_kernel", &mytest_ops);
    //cls = class_create(THIS_MODULE, "thinkforce_kernel_class");
    //mydev = device_create(cls, 0, MKDEV(major,0),NULL,"thinkforce_kernel_device");

    //if (sysfs_create_file(&(mydev->kobj), &dev_attr_kernel_version.attr)) {
    //    return -1;
    //}

    // init mutex for pid list
    mutex_init(&pid_mutex);
    mutex_init(&app_mutex);
    spin_lock_init(&app_spin_lock);
    init_pids();

    // init app lock records
    initRecords();

    {
        unsigned int* periscfg = ioremap(TFACC0_FULL_ACP_BASE, TFACC_CLK_LENGTH);
        retval = periscfg[0x34/4];
        DPRINTK("Init Flag = %llx\n", (unsigned long long)periscfg[0x34/4]);
        if (retval == 0) {
            periscfg[0x34/4] = 1;
            DPRINTK("Flag = %llx\n", (unsigned long long)periscfg[0x34/4]);
            skipFullSwRst = 0;
        } else {
            DPRINTK("Skip Full Cache rst\n");
            skipFullSwRst = retval;
        }
        iounmap(periscfg);
    }

    for (c = 0; c < chips; c++) {
        DPRINTK("init chip %d\n", c);
        gap = chipGap * c;

        tfacc_full_enable(gap + TFACC_BL_CLK_BASE);
        tfacc_full_enable(gap + TFACC_BR_CLK_BASE);
        tfacc_lite_enable(gap + TFACC_L_CLK_BASE);
        tfacc_lite_enable(gap + TFACC_R_CLK_BASE);

        tfacc_full_acp(gap + TFACC0_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC1_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC2_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC3_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));

        if (!skipFullSwRst) {
            tfacc_full_enable_cache(gap + TFACC0_BASE, gap + TFACC1_BASE, gap + TFACC0_CACHE_BASE, gap + TFACC1_CACHE_BASE, gap + TFACC_BL_CLK_BASE, gap+TFACC0_FULL_ACP_BASE);
            tfacc_full_enable_cache(gap + TFACC2_BASE, gap + TFACC3_BASE, gap + TFACC2_CACHE_BASE, gap + TFACC3_CACHE_BASE, gap + TFACC_BR_CLK_BASE, gap+TFACC1_FULL_ACP_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE0_BASE, gap + TFACCLITE0_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE1_BASE, gap + TFACCLITE1_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE2_BASE, gap + TFACCLITE2_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE3_BASE, gap + TFACCLITE3_CACHE_BASE);
        }


        output_tfacc_id(gap + 0xFC000000);
        output_tfacc_id(gap + 0xFC100000);
        output_tfacc_id(gap + 0xEC000000);
        output_tfacc_id(gap + 0xEC100000);
        output_tfacc_id(gap + 0xF9800000);
        output_tfacc_id(gap + 0xF9900000);
        output_tfacc_id(gap + 0xE9800000);
        output_tfacc_id(gap + 0xE9900000);
    }

    if (IS_ERR(thinkforce_class = class_create(THIS_MODULE, "thinkforce_class"))) {
        DPRINTK("failed to device register class\n");
        retval = -ENOMEM;
        goto origin;
    }

    dev = tf_create_and_init_device(0);
    if (dev == NULL) {
        goto origin;
    }

    if (tf_create_and_init_cdev(dev, 0) < 0) {
        goto create_error;
    }

    DPRINTK("EXIT, succeed\n");
    return 0;

    create_error:
#ifdef CONFIG_ACPI
    tf_cleanup_module(pdev);
#else
    tf_cleanup_module();
#endif

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);

    return retval;
}

void tfacc_cache_debug(unsigned long long BASE) {
    volatile unsigned int *apb = ioremap(BASE, DEVICE_IO_LENGTH);
    int i=0;
    if (efuse) {
        return;
    }
    while (i<0x200/4) {
        DPRINTK("%08x = %llx\n", 4*i, (unsigned long long)(*(apb+i)));
        i++;
    }
    iounmap(apb);
}

#ifdef CONFIG_ACPI
static int __exit tf_cleanup_module(struct platform_device *pdev) {
    int c;
    unsigned long long gap;
	if (efuse) {
		return -1;
	}
#else
static void __exit tf_cleanup_module(void) {
    int c;
    unsigned long long gap;
    if (efuse) {
        return;
    }
#endif

    DPRINTK("ENTER\n");

    tf_remove();
    tf_dev = NULL;
    class_destroy(thinkforce_class);

//    device_destroy(cls, MKDEV(major,0));
//    class_destroy(cls);
    unregister_chrdev(major, "mytest");

    for (c = 0; c < chips; c++) {
        DPRINTK("disable chip %d tfacc\n", c);
        gap = chipGap * c;

//        tfacc_cache_debug(gap + TFACC1_CACHE_BASE);
//        tfacc_cache_debug(gap + TFACC3_CACHE_BASE);

        tfacc_full_disable(gap + TFACC_BL_CLK_BASE);
        tfacc_full_disable(gap + TFACC_BR_CLK_BASE);
        tfacc_lite_disable(gap + TFACC_L_CLK_BASE);
        tfacc_lite_disable(gap + TFACC_R_CLK_BASE);
    }
    tfacc_full_checkrstcond();

    DPRINTK("EXIT, succeed\n");
#ifdef CONFIG_ACPI
    return 0;
#endif
}

module_param(needReset, int, S_IWUSR|S_IRUSR);
MODULE_PARM_DESC(needReset, "Enforce TFACC hardware reset");