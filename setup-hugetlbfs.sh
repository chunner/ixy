#!/bin/bash
mkdir -p /mnt/huge
# 原来的：
# (mount | grep /mnt/huge) > /dev/null || mount -t hugetlbfs hugetlbfs /mnt/huge

# 修改后的（显式指定 pagesize=2M）：
(mount | grep /mnt/huge) > /dev/null || mount -t hugetlbfs -o pagesize=2M hugetlbfs /mnt/huge
for i in /sys/devices/system/node/node[0-9]*
do
	echo 512 > "$i"/hugepages/hugepages-2048kB/nr_hugepages
done
