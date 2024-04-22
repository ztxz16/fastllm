// driver for TFDriver inference acceleration
// created by Jingzhou Ji

#ifndef __TFDRIVER_H__
#define __TFDRIVER_H__
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/mii.h>
#include <linux/cdev.h>
#include <linux/dma-mapping.h>
#include <linux/kobject.h>
#include <linux/vmalloc.h>
#include <linux/slab.h>
#include <linux/ioctl.h>
#include <linux/hashtable.h>
#include <linux/delay.h>
#include <asm/cacheflush.h>
#include <linux/platform_device.h>
#include <linux/acpi.h>
//#include <asm/outercache.h>

#define DRIVER_VERSION          "0.4.0"
#ifdef CONFIG_ACPI
#define MODNAME                 "tfacc2_ACPI"
#else
#define MODNAME                 "tfacc2"
#endif
#define DRIVER_LOAD_MSG	        "Think-Force AI driver " DRIVER_VERSION " loaded"
#define DRIVER

MODULE_AUTHOR("think-force.com");
MODULE_DESCRIPTION("Think Force AI Driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(DRIVER_VERSION);

#define REG_CHICKEN_BASE 0xff
#define TF_BUF_HASHBITS 16  //2^15 buf
#define TF_BUF_HASHENTRIES (1 << TF_BUF_HASHBITS)
#define TF_BUF_HASHMASK (TF_BUF_HASHENTRIES - 1)

#define TF_DEBUG

#ifdef TF_DEBUG
#define DPRINTK(fmt, args...) printk(KERN_DEBUG "%s: " fmt, __func__ , ## args)
#else
#define DPRINTK(fmt, args...) do {} while (0)
#endif
#define NDPRINTK(fmt, args...) do {} while(0)

#ifdef TF_DEBUG
#define assert(expr)						\
    if (!(expr)) {						\
        printk("Assertion failed! %s,%s,%s,line=%d\n",	\
               #expr, __FILE__, __func__, __LINE__);	\
    }
#else
#define assert(expr) do {} while (0)
#endif

#ifdef TF_DEBUG
#define DASSERT2(expr1, expr2)						\
    if (expr1 != expr2) {						\
        printk("Assertion failed! %s,%016lx,%s,%016lx,%s,%s,line=%d\n",	\
               #expr1, expr1, #expr2, expr2, __FILE__, __func__, __LINE__);	\
    }
#else
#define DASSERT2(expr1, expr2) do {} while (0)
#endif


static int tf_open(struct inode *inode, struct file *filp);
static int tf_release(struct inode *inode, struct file *filp);
static int tf_mmap(struct file *filp, struct vm_area_struct *vma);
static long tf_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

void tf_remove(void);

#ifdef CONFIG_ACPI
static int __init tf_init_module(struct platform_device *pdev);
static int __exit tf_cleanup_module(struct platform_device *pdev);
#else
static int __init tf_init_module(void);
static void __exit tf_cleanup_module(void);
#endif

#define TF_MAGIC 'x'
#define TF_MAX_NR 9
#define TF_BUF_CLEAR        _IO(TF_MAGIC, 0)
#define TF_BUF_RESET        _IO(TF_MAGIC, 8)
#define TF_BUF_CREATE       _IOWR(TF_MAGIC, 0, struct tf_buf_io_param *)
#define TF_BUF_FLUSHL1      _IOWR(TF_MAGIC, 1, struct tf_get_phy_value *)
#define TF_VERSION_CHECK    _IOWR(TF_MAGIC, 2, struct tf_version *)

// get lock, return 0 as SUCCESS,
// 第一个参数 ， 表示锁的方式：-1: try lock ,
//                         > 0 wait lock. 尝试一段时间, 单位us
//                         = 0 wait until lock，一直等
// 第二个参数，  表示对应锁住的TFACC ID
#define TF_APP_LOCK                _IOW(TF_MAGIC, 3, int*)

// unlock, return 0 as SUCCESS, 如果没有持有锁，解锁也会返回成功
// 第一个参数，  表示对应解锁的TFACC ID
#define TF_APP_UNLOCK              _IOW(TF_MAGIC, 4, int*)

// get pids 正在使用这个driver的，返回pid 列表，-1表示结尾
#define TF_READ_PIDS               _IOR(TF_MAGIC, 5, int*)
#define TF_READ_APP_LOCK_RECORD    _IOR(TF_MAGIC, 6, struct tf_lock_record*)     // 获得锁的历史访问记录
#define TF_READ_RESERVE_MEM_RECORD _IOR(TF_MAGIC, 7, struct ReserveDDRBlock*)    // reserve 内存
#define TF_READ_APP_USAGE            _IOR(TF_MAGIC, 8, int*)  // 获取使用情况

#define TFACC_REG_CNT 						8

#define EFUSE_BASE 0xFE1A046C // EFUSE

#define TFACC0_BASE							0xFC000000 		// TFACC0
#define TFACC1_BASE							0xFC100000		// TFACC1
#define TFACC0_CACHE_BASE					0xFC200000 		// TFACC2 Cache
#define TFACC1_CACHE_BASE					0xFC600000 		// TFACC3 Cache

#define TFACC2_BASE							0xEC000000		// TFACC2
#define TFACC3_BASE							0xEC100000		// TFACC3
#define TFACC2_CACHE_BASE					0xEC200000 		// TFACC2 Cache
#define TFACC3_CACHE_BASE					0xEC600000 		// TFACC3 Cache

#define TFACCLITE0_BASE						0xF9800000		// TFACC Lite 0
#define TFACCLITE1_BASE						0xF9900000		// TFACC Lite 1
#define TFACCLITE0_CACHE_BASE				0xF9000000		// TFACC Lite 1
#define TFACCLITE1_CACHE_BASE				0xF9400000		// TFACC Lite 1

#define TFACCLITE2_BASE						0xE9800000		// TFACC Lite 2
#define TFACCLITE3_BASE						0xE9900000		// TFACC Lite 3
#define TFACCLITE2_CACHE_BASE				0xE9000000		// TFACC Lite 2
#define TFACCLITE3_CACHE_BASE				0xE9400000		// TFACC Lite 3

#define DEVICE_IO_LENGTH    				0x100000

#define TFACC_BL_CLK_BASE                   0xFE110000
#define TFACC_BR_CLK_BASE                   0xED400000
#define TFACC_L_CLK_BASE                    0xB1500000
#define TFACC_R_CLK_BASE                    0xB1100000
#define TFACC_CLK_LENGTH					0x8000

#define TFACC0_FULL_ACP_BASE                 0xFE170000
#define TFACC1_FULL_ACP_BASE                 0xED090000
#define TFACC2_FULL_ACP_BASE                 0xB1460000
#define TFACC3_FULL_ACP_BASE                 0xB10D0000

#define SRAM1ID 10001
#define SRAM2ID 10002
#define REG2ID 10003
#define CACHEREGID 10100
#define REGMAINID 9998

#define MAX_TFACC_CNT 32
#define CBUF 9997

////@brief kernel buf
struct kbuf {
    struct hlist_node list;
    struct kobject kobj;
    void * kernel_addr;
    dma_addr_t phy_addr;
    int len;
    int mmap_id;
};

////@brief device
struct tf_device {
    spinlock_t lock;
    void *ioreg[TFACC_REG_CNT * 10];
    void *ioreg_cache[TFACC_REG_CNT * 10];
    void *ioregMain;
    void *sram1;
    void *sram2;
    unsigned int major;
    unsigned int minor;
    struct cdev cdev;
    struct device *device;
    int isBusy;
    int version;
    int nParallel;
    int nBuf;

    struct kobject kobj;
    struct kbuf reg_buf[TFACC_REG_CNT * 10];
    struct kbuf cache_reg_buf[TFACC_REG_CNT * 10];
    struct kbuf regMain_buf;
    struct kbuf sram1_buf;
    struct kbuf sram2_buf;
    struct kbuf cbuf; // cachable buf
    struct kobject buf_list_top;
    DECLARE_HASHTABLE(buf_list, TF_BUF_HASHBITS);

    int mmap_id_counter;
    int device_id;

    int useDDR2;         // 7140之后的芯片，这个属性用来表示分配在哪个numa节点上
    // int holdAppLockPid;  // 当前持有app锁的pid
    // int holdAppLockTgid; // 当前持有app锁的tgid

    /// 每个TFACC目前被哪个进程占用
    int holdTFACCPid[MAX_TFACC_CNT];
    /// 每个TFACC目前被哪个TGID占用
    int holdTFACCTgid[MAX_TFACC_CNT];
};

struct tf_buf_io_param{
    dma_addr_t phy_addr;
    int len;
    int mmap_id;
    int uncache;
    int useDDR2;
};

struct tf_version {
    int sdk_version; //传过来SDK编号，检测该SDK是否可以运行在这个Driver上
    int kernel_version; //传回去的Kernel编号，在SDK端检测Kernel编号是否符合要求
    int excepted_sdk_version; //当SDK版本过低时，以此来表示至少需要的SDK版本
};

struct tf_get_phy_value {
    unsigned int phy_addr;
    int len;
    unsigned int *dst;
};

struct tf_app_info {
    int pid;                      // 打开driver的pid
    int tgid;                     // thread group id
};

// Parameter 'size' for an ioctl code is limited with (16K -1).
// 512 * (4 + 4 + 8 + 8 + 4 + 4) = 16K
struct tf_lock_record {
    int pid;                       // 持有app_lock 的pid
    int tgid;                      // 持有app_lock 的tid
    unsigned long lockTime;        // jiffies
    unsigned long unlockTime;      // jiffies, 如果正在持有, jiffies = 最新
    int isHolding;                 // 是否还在持有
    int tfaccID;                   // 锁的TFACC ID
};

// reserve 内存分配信息, size = 8 + 8 + 8 + 4 + 1 + 4 = 33
struct ReserveDDRBlock {
    long long startPos;            // 起始地址
    long long offset;              // 当前
    long long len;                 // 最大长度
    int chipId;                    // 归属的芯片ID
    bool isMalloc;                 // 是否被使用
    int tgid;                      // 线程组id
};

// 
static const struct file_operations tf_device_ops = {
        .owner          = THIS_MODULE,
        .open           = tf_open,
        .release        = tf_release,
        .mmap	        = tf_mmap,
        .unlocked_ioctl = tf_ioctl,
};


static struct class *thinkforce_class;

#ifdef CONFIG_ACPI
static const struct acpi_device_id acpi_tf_tfacc_match[] = {
	{ "TFA0001" },
	{  }
};
MODULE_DEVICE_TABLE(acpi, acpi_tf_tfacc_match);

static struct platform_driver tf_tfacc_driver = {
    .driver = {
        .name           = "tf_tfacc",
        .pm             = NULL,
        .of_match_table = NULL,
        .acpi_match_table = ACPI_PTR(acpi_tf_tfacc_match),
    },
    .probe  = tf_init_module,
    .remove = tf_cleanup_module,
};
module_platform_driver(tf_tfacc_driver);
#else
module_init(tf_init_module);
module_exit(tf_cleanup_module);
#endif

#endif // PCI_THINKFORCE_H
