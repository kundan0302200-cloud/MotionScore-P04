/*
 * MotionScore Edge — NUCLEO-F446RE + LM35 telemetry demo.
 *
 * CubeMX: ADC1 channel 0 on PA0, 12-bit/right aligned/software trigger.
 * USART2: PA2 TX / PA3 RX, 115200, 8-N-1.
 *
 * Wire protocol: MS1,<HAL tick milliseconds>,<12-bit average ADC raw>\r\n
 * The host PC converts ADC raw count to temperature, so printf float support
 * is not needed in the microcontroller firmware.
 *
 * This monitors ambient temperature only. There is no actuator and this is not
 * closed-loop temperature control.
 */
#include "motionscore_edge.h"
#include "adc.h"
#include "usart.h"
#include "stm32f4xx_hal.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MS_ADC_AVERAGE_SAMPLES 16U
#define MS_REPORT_INTERVAL_MS 500U
#define MS_ADC_TIMEOUT_MS 20U
#define MS_UART_TIMEOUT_MS 100U

static uint32_t next_report_tick = 0U;

static bool MotionScoreEdge_ReadAdcAverage(uint16_t *average)
{
    uint32_t sum = 0U;
    if (average == NULL) {
        return false;
    }

    for (uint32_t i = 0U; i < MS_ADC_AVERAGE_SAMPLES; ++i) {
        if (HAL_ADC_Start(&hadc1) != HAL_OK) {
            return false;
        }
        if (HAL_ADC_PollForConversion(&hadc1, MS_ADC_TIMEOUT_MS) != HAL_OK) {
            (void)HAL_ADC_Stop(&hadc1);
            return false;
        }
        sum += HAL_ADC_GetValue(&hadc1);
        if (HAL_ADC_Stop(&hadc1) != HAL_OK) {
            return false;
        }
    }
    *average = (uint16_t)(sum / MS_ADC_AVERAGE_SAMPLES);
    return true;
}

void MotionScoreEdge_Init(void)
{
    next_report_tick = HAL_GetTick();
}

void MotionScoreEdge_Loop(void)
{
    const uint32_t now = HAL_GetTick();
    uint16_t adc_average = 0U;
    char frame[48];
    int frame_length;

    if ((int32_t)(now - next_report_tick) < 0) {
        return;
    }
    next_report_tick = now + MS_REPORT_INTERVAL_MS;

    if (!MotionScoreEdge_ReadAdcAverage(&adc_average)) {
        return;
    }

    frame_length = snprintf(frame, sizeof(frame), "MS1,%lu,%u\r\n",
                            (unsigned long)HAL_GetTick(),
                            (unsigned int)adc_average);
    if (frame_length <= 0 || (size_t)frame_length >= sizeof(frame)) {
        return;
    }

    (void)HAL_UART_Transmit(&huart2, (uint8_t *)frame,
                            (uint16_t)frame_length, MS_UART_TIMEOUT_MS);
}
