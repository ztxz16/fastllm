obj-m += tfacc2.o
PWD=$(shell pwd)

#LINUX_VERSION=$(LINUX_VERSION)
KERNEL_VERSION=$(shell uname -r)

ifeq ($(LINUX_VERSION),Ubuntu)
	KDIR=/usr/src/linux-headers-$(KERNEL_VERSION)
else
	KDIR=/usr/src/kernels/$(KERNEL_VERSION)/
endif


install:
	bash ./build_driver.sh
	
tfacc2:
	make -C $(KDIR) M=$(PWD) modules
	make -C $(KDIR) M=$(PWD) modules_install
	depmod -A
clean:
	make -C $(KDIR) M=$(PWD) clean
