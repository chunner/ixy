#include <stdio.h>

#include "stats.h"
#include "log.h"
#include "memory.h"
#include "driver/device.h"
#include "driver/cdma.h"
#include "ixy-pktgen.h"

// number of packets sent simultaneously to our driver
// static const uint32_t BATCH_SIZE = 1;  // 64;

// excluding CRC (offloaded by default)
#define PKT_SIZE 60

static const uint8_t pkt_data[] = {
	0x01, 0x02, 0x03, 0x04, 0x05, 0x06, // dst MAC
	0x10, 0x10, 0x10, 0x10, 0x10, 0x10, // src MAC
	0x08, 0x00,                         // ether type: IPv4
	0x45, 0x00,                         // Version, IHL, TOS
	(PKT_SIZE - 14) >> 8,               // ip len excluding ethernet, high byte
	(PKT_SIZE - 14) & 0xFF,             // ip len exlucding ethernet, low byte
	0x00, 0x00, 0x00, 0x00,             // id, flags, fragmentation
	0x40, 0x11, 0x00, 0x00,             // TTL (64), protocol (UDP), checksum
	0x0A, 0x00, 0x00, 0x01,             // src ip (10.0.0.1)
	0x0A, 0x00, 0x00, 0x02,             // dst ip (10.0.0.2)
	0x00, 0x2A, 0x05, 0x39,             // src and dst ports (42 -> 1337)
	(PKT_SIZE - 20 - 14) >> 8,          // udp len excluding ip & ethernet, high byte
	(PKT_SIZE - 20 - 14) & 0xFF,        // udp len exlucding ip & ethernet, low byte
	0x00, 0x00,                         // udp checksum, optional
	'i', 'x', 'y'                       // payload
	// rest of the payload is zero-filled because mempools guarantee empty bufs
};

// calculate a IP/TCP/UDP checksum
static uint16_t calc_ip_checksum(uint8_t *data, uint32_t len) {
	//fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
	if (len % 1) error("odd-sized checksums NYI"); // we don't need that
	uint32_t cs = 0;
	for (uint32_t i = 0; i < len / 2; i++) {
		cs += ((uint16_t *) data)[i];
		if (cs > 0xFFFF) {
			cs = (cs & 0xFFFF) + 1; // 16 bit one's complement
		}
	}
	return ~((uint16_t) cs);
}

static struct mempool *init_mempool() {
	fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
	const int NUM_BUFS = 2048;
	struct mempool *mempool = memory_allocate_mempool(NUM_BUFS, 0);
	// pre-fill all our packet buffers with some templates that can be modified later
	// we have to do it like this because sending is async in the hardware; we cannot re-use a buffer immediately
	struct pkt_buf *bufs[NUM_BUFS];
	for (int buf_id = 0; buf_id < NUM_BUFS; buf_id++) {
		struct pkt_buf *buf = pkt_buf_alloc(mempool);
		buf->size = PKT_SIZE;
		memcpy(buf->data, pkt_data, sizeof(pkt_data));
		*(uint16_t *) (buf->data + 24) = calc_ip_checksum(buf->data + 14, 20);
		bufs[buf_id] = buf;
	}
	// return them all to the mempool, all future allocations will return bufs with the data set above
	for (int buf_id = 0; buf_id < NUM_BUFS; buf_id++) {
		pkt_buf_free(bufs[buf_id]);
	}

	return mempool;
}

int main(int argc, char *argv[]) {
	fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
	if (argc != 2) {
		printf("Usage: %s <pci bus id>\n", argv[0]);
		return 1;
	}

	struct ixy_device *dev = ixy_init(argv[1], 1, 1, 0);
	struct cdma_device *cdma_dev = IXY_TO_CDMA(dev);
	struct mempool *mempool = init_mempool();

	printf("=== CDMA Test ===\n");

	uint32_t seq_num = 0;

	// array of bufs sent out in a batch
	// struct pkt_buf* bufs[BATCH_SIZE];
	// struct pkt_buf *bufs = pkt_buf_alloc(mempool);
	// struct pkt_buf *bufs_dst = pkt_buf_alloc(mempool);
	struct dma_memory bufs = memory_allocate_dma(PKT_SIZE, true);
	struct dma_memory bufs_dst = memory_allocate_dma(PKT_SIZE, true);

	debug("Source buffer: 		virt=0x%p, phy=0x%lx\n", bufs.virt, bufs.phy);
	debug("Destination buffer:	virt=0x%p, phy=0x%lx\n", bufs_dst.virt, bufs_dst.phy);



	cdma_dev->default_dst_mem.phy = bufs_dst.phy;
	cdma_dev->default_dst_mem.virt = bufs_dst.virt;


	*(uint32_t *) (bufs.virt) = 0x12345678;
	*(uint32_t *) (bufs.virt + PKT_SIZE - 4) = 0xabcdef01;
	*(uint32_t *) (bufs_dst.virt) = 0;
	*(uint32_t *) (bufs_dst.virt + PKT_SIZE - 4) = 0;
	debug("Before CDMA transfer:\n");
	debug("source , %x, %x", *(uint32_t *) (bufs.virt), *(uint32_t *) (bufs.virt + PKT_SIZE - 4));
	debug("destination , %x, %x", *(uint32_t *) (bufs_dst.virt), *(uint32_t *) (bufs_dst.virt + PKT_SIZE - 4));
	// ixy_tx_batch_busy_wait(dev, 0, &bufs, 1);
	cdma_simple_transfer(cdma_dev, bufs.phy, bufs_dst.phy, PKT_SIZE);
	// check
	debug("After CDMA transfer:\n");
	debug("source , %x, %x", *(uint32_t *) (bufs.virt), *(uint32_t *) (bufs.virt + PKT_SIZE - 4));
	debug("destination , %x, %x", *(uint32_t *) (bufs_dst.virt), *(uint32_t *) (bufs_dst.virt + PKT_SIZE - 4));


	return 0;
}


int data_mover(const char *pci_addr, uint64_t src_addr, uint64_t dst_addr, uint32_t size) {
	fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
	struct ixy_device *dev = ixy_init(pci_addr, 1, 1, 0);
	struct cdma_device *cdma_dev = IXY_TO_CDMA(dev);


	uint64_t src_phy = virt_to_phys((void *) src_addr);
	uint64_t dst_phy = virt_to_phys((void *) dst_addr);
	cdma_simple_transfer(cdma_dev, src_phy, dst_phy, size);

	return 0;
}
int execute(const char *pci_addr, uint64_t src_addr, uint64_t dst_addr, uint32_t size) {
	fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
	struct ixy_device *dev = ixy_init(pci_addr, 1, 1, 0);
	struct cdma_device *cdma_dev = IXY_TO_CDMA(dev);


	uint64_t src_phy = virt_to_phys((void *) src_addr);
	uint64_t role_phy = (void *) 0x8000000000000000;
	uint64_t dst_phy = virt_to_phys((void *) dst_addr);
	cdma_simple_transfer(cdma_dev, src_phy, role_phy, size);
	cdma_simple_transfer(cdma_dev, role_phy, dst_phy, size);

	return 0;
}