/*
 * AXI CDMA v4.1 Register Definitions and Data Structures
 * 
 * Based on Xilinx PG034 - AXI Central Direct Memory Access v4.1
 * Product Guide
 */

#ifndef _AXI_CDMA_TYPE_H_
#define _AXI_CDMA_TYPE_H_

#include <stdint.h>
#include "device.h"
#include "memory.h"


/* AXI CDMA Register Offsets (relative to BAR0 base address) */
#define CDMACR          0x00    /* CDMA Control Register */
#define CDMASR          0x04    /* CDMA Status Register */
#define CURDESC_PNTR    0x08    /* Current Descriptor Pointer Register */
#define CURDESC_PNTR_MSB 0x0C   /* Current Descriptor Pointer Upper 32 bits */
#define TAILDESC_PNTR   0x10    /* Tail Descriptor Pointer Register */
#define TAILDESC_PNTR_MSB 0x14  /* Tail Descriptor Pointer Upper 32 bits */
#define SA              0x18    /* Source Address Register */
#define SA_MSB          0x1C    /* Source Address Upper 32 bits */
#define DA              0x20    /* Destination Address Register */
#define DA_MSB          0x24    /* Destination Address Upper 32 bits */
#define BTT             0x28    /* Bytes to Transfer Register */

/* CDMA Control Register (CDMACR) bit definitions */
#define CDMACR_TAIL_PNTR_EN     (1 << 1)   /* Tail Pointer Enable */
#define CDMACR_RESET            (1 << 2)   /* Soft Reset */
#define CDMACR_SGMode           (1 << 3)   /* Scatter Gather Mode Enable */
#define CDMACR_KeyHole_Read     (1 << 4)   /* Key Hole Read Enable */
#define CDMACR_KeyHole_Write    (1 << 5)   /* Key Hole Write Enable */
#define CDMACR_Cyclic_BD_Enable (1 << 6)   /* Cyclic Buffer Descriptor Enable */
#define CDMACR_IOC_IrqEn        (1 << 12)  /* Interrupt on Complete Enable */
#define CDMACR_Dly_IrqEn        (1 << 13)  /* Interrupt on Delay Enable */
#define CDMACR_Err_IrqEn        (1 << 14)  /* Interrupt on Error Enable */
#define CDMACR_IRQThreshold     (0xFF << 16) /* Interrupt Threshold */
#define CDMACR_IRQDelay         (0xFF << 24) /* Interrupt Delay Timer */

/* CDMA Status Register (CDMASR) bit definitions */
#define CDMASR_Idle             (1 << 1)   /* CDMA Idle */
#define CDMASR_SGIncld          (1 << 3)   /* Scatter Gather Included */
#define CDMASR_DMAIntErr        (1 << 4)   /* DMA Internal Error */
#define CDMASR_DMASlvErr        (1 << 5)   /* DMA Slave Error */
#define CDMASR_DMADecErr        (1 << 6)   /* DMA Decode Error */
#define CDMASR_SGIntErr         (1 << 8)   /* Scatter Gather Internal Error */
#define CDMASR_SGSlvErr         (1 << 9)   /* Scatter Gather Slave Error */
#define CDMASR_SGDecErr         (1 << 10)  /* Scatter Gather Decode Error */
#define CDMASR_IOC_Irq          (1 << 12)  /* Interrupt on Complete */
#define CDMASR_Dly_Irq          (1 << 13)  /* Delay Interrupt */
#define CDMASR_Err_Irq          (1 << 14)  /* Error Interrupt */
#define CDMASR_IRQThresholdSts  (0xFF << 16) /* Interrupt Threshold Status */
#define CDMASR_IRQDelaySts      (0xFF << 24) /* Interrupt Delay Status */

/* Error masks for easy checking */
#define CDMASR_ALL_ERR_MASK     (CDMASR_DMAIntErr | CDMASR_DMASlvErr | \
                                CDMASR_DMADecErr | CDMASR_SGIntErr | \
                                CDMASR_SGSlvErr | CDMASR_SGDecErr)

/* Maximum transfer sizes */
#define CDMA_MAX_SIMPLE_TRANSFER_SIZE   0x3FFFFFF   /* 64MB - 1 for simple mode */
#define CDMA_MAX_SG_TRANSFER_SIZE       0x3FFFFFF   /* 64MB - 1 for SG mode */

// /* Buffer Descriptor flags for Scatter Gather mode */
// #define BD_CTRL_TXSOF           (1 << 27)  /* Start of Frame */
// #define BD_CTRL_TXEOF           (1 << 26)  /* End of Frame */
// #define BD_STS_COMPLETE_MASK    (1 << 31)  /* Completed */
// #define BD_STS_DEC_ERR_MASK     (1 << 30)  /* Decode Error */
// #define BD_STS_SLV_ERR_MASK     (1 << 29)  /* Slave Error */
// #define BD_STS_INT_ERR_MASK     (1 << 28)  /* Internal Error */
// #define BD_STS_ALL_ERR_MASK     (BD_STS_DEC_ERR_MASK | BD_STS_SLV_ERR_MASK | 
//                                 BD_STS_INT_ERR_MASK)

// /**
//  * AXI CDMA Buffer Descriptor structure for Scatter Gather mode
//  * Must be aligned to cache line boundary (typically 64 bytes)
//  */
// struct axi_cdma_bd {
//     uint32_t next_desc;         /* Next Descriptor Pointer (lower 32 bits) */
//     uint32_t next_desc_msb;     /* Next Descriptor Pointer (upper 32 bits) */
//     uint32_t src_addr;          /* Source Address (lower 32 bits) */
//     uint32_t src_addr_msb;      /* Source Address (upper 32 bits) */
//     uint32_t dest_addr;         /* Destination Address (lower 32 bits) */  
//     uint32_t dest_addr_msb;     /* Destination Address (upper 32 bits) */
//     uint32_t control;           /* Control field */
//     uint32_t status;            /* Status field */
//     uint32_t reserved[8];       /* Reserved for alignment to 64 bytes */
// } __attribute__((packed, aligned(64)));

// /**
//  * Simple transfer descriptor for non-SG mode
//  */
// struct axi_cdma_simple_transfer {
//     uint64_t src_addr;          /* Source address (64-bit) */
//     uint64_t dest_addr;         /* Destination address (64-bit) */
//     uint32_t length;            /* Transfer length in bytes */
//     uint32_t flags;             /* Transfer flags */
// };

// /**
//  * CDMA queue structure for managing transfers
//  */
// struct axi_cdma_queue {
//     struct axi_cdma_bd* bd_ring;        /* Buffer descriptor ring */
//     uint32_t bd_ring_phys_addr;         /* Physical address of BD ring */
//     uint32_t bd_ring_size;              /* Number of BDs in ring */
//     uint32_t bd_head;                   /* Head index */
//     uint32_t bd_tail;                   /* Tail index */
//     uint32_t bd_free_count;             /* Free BD count */
//     struct mempool* mempool;            /* Memory pool for buffers */
// };

/**
 * Main CDMA device structure
 */
// /* Helper macros for register access */
// #define cdma_read_reg(dev, offset) 
//     (*(volatile uint32_t*)((uint8_t*)(dev)->base_addr + (offset)))

// #define cdma_write_reg(dev, offset, value) 
//     (*(volatile uint32_t*)((uint8_t*)(dev)->base_addr + (offset)) = (value))

// /* Utility functions for 64-bit address handling */
// static inline void cdma_set_src_addr(struct cdma_device* dev, uint64_t addr) {
//     cdma_write_reg(dev, SA, (uint32_t)(addr & 0xFFFFFFFF));
//     if (dev->config.addr_width == 64) {
//         cdma_write_reg(dev, SA_MSB, (uint32_t)(addr >> 32));
//     }
// }

// static inline void cdma_set_dest_addr(struct cdma_device* dev, uint64_t addr) {
//     cdma_write_reg(dev, DA, (uint32_t)(addr & 0xFFFFFFFF));
//     if (dev->config.addr_width == 64) {
//         cdma_write_reg(dev, DA_MSB, (uint32_t)(addr >> 32));
//     }
// }

// static inline uint32_t cdma_get_status(struct cdma_device* dev) {
//     return cdma_read_reg(dev, CDMASR);
// }

// static inline void cdma_clear_interrupts(struct cdma_device* dev) {
//     uint32_t status = cdma_read_reg(dev, CDMASR);
//     cdma_write_reg(dev, CDMASR, status & (CDMASR_IOC_Irq | CDMASR_Dly_Irq | CDMASR_Err_Irq));
// }

// /* Status checking functions */
// static inline bool cdma_is_idle(struct cdma_device* dev) {
//     return (cdma_get_status(dev) & CDMASR_Idle) != 0;
// }

// static inline bool cdma_is_halted(struct cdma_device* dev) {
//     return (cdma_get_status(dev) & CDMASR_Halted) != 0;
// }

// static inline bool cdma_has_error(struct cdma_device* dev) {
//     return (cdma_get_status(dev) & CDMASR_ALL_ERR_MASK) != 0;
// }

// /* Transfer alignment requirements */
// #define CDMA_ALIGNMENT_BYTES    64      /* Recommended alignment for optimal performance */
// #define CDMA_MIN_ALIGNMENT      4       /* Minimum required alignment */

/* Convert ixy_device to cdma_device */

#endif /* _AXI_CDMA_TYPE_H_ */