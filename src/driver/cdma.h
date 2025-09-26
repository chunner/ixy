#ifndef CDMA_H
#define CDMA_H
#include <stdbool.h>
#include "stats.h"
#include "memory.h"

/**
 * CDMA device configuration structure
 */
struct axi_cdma_config {
    uint32_t max_burst_len;     /* Maximum burst length */
    uint32_t data_width;        /* Data width (32, 64, 128, 256, 512, 1024 bits) */
    uint8_t  addr_width;        /* Address width (32 or 64 bits) */
    bool     sg_enabled;        /* Scatter Gather support */
    bool     keyhole_read;      /* Key hole read support */
    bool     keyhole_write;     /* Key hole write support */
    uint8_t  max_outstanding;   /* Maximum outstanding transfers */
};


struct cdma_device {
    struct ixy_device ixy;              /* Base ixy device */

    /* Hardware resources */
    void *base_addr;                    /* BAR0 mapped base address */
    uint16_t device_id;                 /* PCI device ID */

    /* Device configuration */
    struct axi_cdma_config config;      /* Device configuration */

    // struct dma_memory default_src_mem; /* Default source memory */
    struct dma_memory default_dst_mem; /* Default destination memory */

    // /* Queues */
    // struct axi_cdma_queue* tx_queue;    /* Transmit queue */
    // struct axi_cdma_queue* rx_queue;    /* Receive queue */

    // /* Statistics */
    // uint64_t tx_packets;                /* Transmitted packets */
    // uint64_t rx_packets;                /* Received packets */
    // uint64_t tx_bytes;                  /* Transmitted bytes */
    // uint64_t rx_bytes;                  /* Received bytes */
    // uint64_t errors;                    /* Error count */
};
#define IXY_TO_CDMA(ixy_device) container_of(ixy_device, struct cdma_device, ixy);

struct ixy_device *cdma_init(const char *pci_addr, uint16_t rx_queues, uint16_t tx_queues);
uint32_t cmda_tx_batch(struct ixy_device *dev, uint16_t queue_id, struct pkt_buf *bufs[], uint32_t num_bufs);

int cdma_simple_transfer(struct cdma_device *dev, uint64_t src_addr, uint64_t dst_addr, uint32_t length);
#endif // CDMA_H