#include <emmintrin.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#include <linux/limits.h>
#include <linux/vfio.h>
#include <sys/stat.h>

#include "driver/device.h"
#include "log.h"
#include "memory.h"
#include "pci.h"
#include "cdma_type.h"
#include "cdma.h"
#include "libixy-vfio.h"

static const char *driver_name = "ixy-cdma";

uint32_t cmda_tx_batch(struct ixy_device *dev, uint16_t queue_id, struct pkt_buf *bufs[], uint32_t num_bufs);

static void cdma_device_init(struct cdma_device *dev) {
    fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
    // 1. Soft Reset for CDMA 
    debug("Resetting CDMA device");
    set_reg32(dev->base_addr, CDMACR, CDMACR_RESET);
    while ((get_reg32(dev->base_addr, CDMACR) & CDMACR_RESET) != 0) {
        usleep(100);
    }
    uint32_t status = get_reg32(dev->base_addr, CDMACR);
    debug("CDMA reset complete, status: 0x%08x", status);

    // 2. Configure CDMACR
    uint32_t control = 0;
    if (dev->config.sg_enabled) {
        control |= CDMACR_SGMode;
        control |= CDMACR_TAIL_PNTR_EN;
    }
    if (dev->config.keyhole_read) {
        control |= CDMACR_KeyHole_Read;
    }
    if (dev->config.keyhole_write) {
        control |= CDMACR_KeyHole_Write;
    }

    set_reg32(dev->base_addr, CDMACR, control);
    debug("CDMA control register set to 0x%08x", control);

    _mm_mfence();

    info("Setup complete");

}

struct ixy_device *cdma_init(const char *pci_addr, uint16_t rx_queues, uint16_t tx_queues) {
    fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
    if (getuid()) {
        warn("Not running as root, this will probably fail");
    }
    if (rx_queues > 1) {
        error("cannot configure %d rx queues: limit is %d", rx_queues, 1);
    }
    if (tx_queues > 1) {
        error("cannot configure %d tx queues: limit is %d", tx_queues, 1);
    }
    remove_driver(pci_addr);
    struct cdma_device *dev = calloc(1, sizeof(*dev));
    // 1. Initialize the common Ixy device part
    dev->ixy.pci_addr = strdup(pci_addr);
    dev->ixy.driver_name = driver_name;
    dev->ixy.num_rx_queues = rx_queues;
    dev->ixy.num_tx_queues = tx_queues;
    dev->ixy.rx_batch = NULL;
    dev->ixy.tx_batch = cmda_tx_batch;
    dev->ixy.read_stats = NULL;
    dev->ixy.set_promisc = NULL;
    dev->ixy.get_link_speed = NULL;

    // 2. Device Configuration
    dev->config.max_burst_len = 16; // Max burst length of 16
    dev->config.addr_width = 64; // 64-bit addressing
    dev->config.data_width = 32; // 32-bit data width
    dev->config.sg_enabled = false; // Scatter-Gather enabled
    dev->config.keyhole_read = false; // No keyhole read
    dev->config.keyhole_write = false; // No keyhole write
    dev->config.max_outstanding = 1; // Max 1 outstanding transfer

    // 3. Map BAR0 region
    dev->base_addr = pci_map_resource(pci_addr);

    // 4. Read device ID from config space
    int config = pci_open_resource(pci_addr, "config", O_RDONLY);
    uint16_t device_id = read_io16(config, 2); printf("device_id: 0x%04x\n", device_id);
    dev->device_id = device_id;
    close(config);

    // 5. Initialize the CDMA device
    cdma_device_init(dev);
    return &dev->ixy;

}





static int cdma_simple_transfer(struct cdma_device *dev, uint64_t src_addr, uint64_t dst_addr, uint32_t length) {
    fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);

    // 1. Check Parameters
    if (length == 0 || length > CDMA_MAX_SIMPLE_TRANSFER_SIZE) {
        error("Invalid transfer length: %u", length);
        return -1;
    }
    // 2. Wait for Idle
    int timeout = 10000; // 1 second timeout
    while (!(get_reg32(dev->base_addr, CDMASR) & CDMASR_Idle) && timeout > 0) {
        usleep(100);
        timeout--;
    }
    if (timeout == 0) {
        error("CDMA not idle, timeout waiting for idle");
        return -1;
    }
    // 3. Check for Errors
    uint32_t status = get_reg32(dev->base_addr, CDMASR);
    if (status & CDMASR_ALL_ERR_MASK) {
        error("CDMA error detected, status: 0x%08x", status);
        // Clear Errors
        set_reg32(dev->base_addr, CDMASR, status & CDMASR_ALL_ERR_MASK);
        return -1;
    }
    // 4. Set Source and Destination Addresses
    set_reg32(dev->base_addr, SA, (uint32_t) (src_addr & 0xFFFFFFFF));
    if (dev->config.addr_width == 64) {
        set_reg32(dev->base_addr, SA_MSB, (uint32_t) (src_addr >> 32));
    }
    set_reg32(dev->base_addr, DA, (uint32_t) (dst_addr & 0xFFFFFFFF));
    if (dev->config.addr_width == 64) {
        set_reg32(dev->base_addr, DA_MSB, (uint32_t) (dst_addr >> 32));
    }
    _mm_mfence();
    // 5. Set Length and Start Transfer
    set_reg32(dev->base_addr, BTT, length);
    _mm_mfence();
    // 6. Wait for Completion
    while (!(get_reg32(dev->base_addr, CDMASR) & CDMASR_Idle)) {
        usleep(100);
    }
    // 7. Check for Errors Again
    status = get_reg32(dev->base_addr, CDMASR);
    if (status & CDMASR_ALL_ERR_MASK) {
        error("CDMA error detected after transfer, status: 0x%08x", status);
        // Clear Errors
        set_reg32(dev->base_addr, CDMASR, status & CDMASR_ALL_ERR_MASK);
        return -1;
    }
    // Transfer Complete
    debug("CDMA transfer of %u bytes from 0x%016lx to 0x%016lx completed", length, src_addr, dst_addr);
    return 0;
}




// only supports single queue (queue_id 0) and single buffer per batch
uint32_t cmda_tx_batch(struct ixy_device *dev, uint16_t queue_id, struct pkt_buf *bufs[], uint32_t num_bufs) {
    fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
    // Validate parameters
    if (queue_id != 0) { error("Invalid queue ID %d for CDMA device", queue_id);return 0; }
    if (num_bufs == 0) return 0;
    if (num_bufs > 1) { error("CDMA device supports only single buffer per batch, truncating to 1"); return 0; }

    struct cdma_device *cdma_dev = IXY_TO_CDMA(dev);

    struct pkt_buf *buf = bufs[0];

    uint64_t src_addr = buf->buf_addr_phy; // Source address in device memory
    uint64_t dst_addr = cdma_dev->default_dst_mem.phy;

    cdma_simple_transfer(cdma_dev, src_addr, dst_addr, buf->size);
    // Return number of packets received (1 in this case)
    return 1;
}
uint32_t cmda_rx_batch(struct ixy_device *dev, uint16_t queue_id, struct pkt_buf *bufs[], uint32_t num_bufs) {
    fprintf(stdout, "[LOG]: call_stack: %s: %4d: %s\n", __FILE__, __LINE__, __FUNCTION__);
    error("CDMA device does not support RX");
    return 0;
}
