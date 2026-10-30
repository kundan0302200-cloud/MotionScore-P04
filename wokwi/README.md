# Wokwi Edge simulation (NUCLEO-C031C6 + potentiometer)

This is a **simulation-only bridge** into the existing MotionScore P04 model.
Wokwi supports the NUCLEO-C031C6, not the NUCLEO-F446RE. The C031 board is
used here only to exercise ADC acquisition, UART framing and host integration;
the existing F446RE hardware-target firmware remains in embedded/stm32_f446re.

## Data path and meaning

1. Rotate the Wokwi potentiometer connected to the board's A0 ADC.
2. Firmware samples a 12-bit ADC count and emits MS1,<time_ms>,<adc_raw> at
   115200 baud every 500 ms over USART2/VCOM.
3. Python receives it through Wokwi for VS Code's RFC2217 server on port 4000.
4. The potentiometer count maps to a **simulated setpoint parameter** from
   20 to 50. It is NOT an LM35 reading and NOT a temperature in °C.
5. Existing simulate_response() produces a synthetic first-order PI plant run.
   Existing extract_features() creates the same 11 model inputs used in
   training, and the existing Random Forest predicts signed final error.

The input is simulated in Wokwi; the plant response and its reference final
error are simulated in Python. This is a serial-input + model integration
demonstration, not physical-plant validation or true hardware-in-the-loop
control.

## Circuit

Use diagram.json in this folder:

- Potentiometer VCC -> NUCLEO-C031C6 3V3
- Potentiometer GND -> NUCLEO-C031C6 GND
- Potentiometer SIG -> NUCLEO-C031C6 A0
- USART2 PA2/PA3 connects to the board VCOM serial interface.

## Wokwi VS Code setup

Wokwi for VS Code runs a local firmware build. Start from the official sample:
https://github.com/wokwi/stm32-hello-wokwi

1. Install Wokwi for VS Code, activate the Wokwi license, and install the
   STM32 VS Code extension, STM32CubeCLT and Arm GNU Toolchain.
2. Clone wokwi/stm32-hello-wokwi into a separate folder. It is an STM32 HAL
   project for NUCLEO-C031C6.
3. Copy this folder's diagram.json into that sample project root.
4. Open its CubeMX .ioc file and enable ADC1 regular conversion on the board
   A0 / PA0 analog pin. Confirm the exact ADC channel shown by CubeMX. Use
   12-bit resolution, right alignment, software trigger, one conversion.
   Keep USART2 asynchronous at 115200, 8-N-1, PA2 TX / PA3 RX.
5. In generated Core/Src/main.c, declare/configure ADC_HandleTypeDef hadc1
   using CubeMX. Add #include <stdio.h>. Keep the sample's HAL startup and
   clock initialization. Call MX_ADC1_Init() after GPIO and before the loop.
6. Add the following sampling code inside while(1). This assumes CubeMX has
   generated hadc1 and huart2. Adapt names only if your generated handles differ.
7. Copy wokwi.toml.example to the sample root as wokwi.toml. Build with the
   sample's documented make command and ensure its ELF/HEX outputs match
   the paths in wokwi.toml.
8. In VS Code run **Wokwi: Start Simulator**. Keep the simulator tab visible;
   the simulation may pause when its tab is hidden. Python uses the RFC2217
   endpoint, not the browser Wokwi Serial Monitor.

### Firmware loop fragment

Add #include <stdio.h> to main.c, generate/configure hadc1 in CubeMX, then
put this inside while(1):

```c
uint32_t sum = 0U;
uint8_t valid = 1U;
for (uint32_t i = 0U; i < 16U; ++i) {
    if (HAL_ADC_Start(&hadc1) != HAL_OK ||
        HAL_ADC_PollForConversion(&hadc1, 20U) != HAL_OK) {
        (void)HAL_ADC_Stop(&hadc1);
        valid = 0U;
        break;
    }
    sum += HAL_ADC_GetValue(&hadc1);
    (void)HAL_ADC_Stop(&hadc1);
}
if (valid) {
    uint16_t average = (uint16_t)(sum / 16U);
    char frame[48];
    int n = snprintf(frame, sizeof(frame), "MS1,%lu,%u\r\n",
                     (unsigned long)HAL_GetTick(), (unsigned int)average);
    if (n > 0 && (size_t)n < sizeof(frame)) {
        (void)HAL_UART_Transmit(&huart2, (uint8_t *)frame,
                                (uint16_t)n, 100U);
    }
}
HAL_Delay(500U);
```

This is an integration fragment, not a complete standalone main.c. Do not
copy it into the existing F446RE project; it uses the C031 sample's HAL family.

## Python and dashboard

From the MotionScore project root:

```powershell
python -m pip install -r requirements.txt
python train.py                 # only if models/steady_state_model.joblib is absent
python -m streamlit run app.py
```

Select **Wokwi Simulation** in the dashboard sidebar. Leave the URL as
rfc2217://localhost:4000 and click **Capture Wokwi predictions**. Move the
potentiometer during the capture. The results table and CSV show raw ADC,
mapped simulated setpoint, Random Forest predicted signed error/output, and
the synthetic reference error.

For a terminal stream instead:

```powershell
python -m motionscore.wokwi_edge --url rfc2217://localhost:4000 --baud 115200
```

## Correct interpretation for a demo/review

- ADC 0–4095 is the potentiometer position as a digital count, not temperature.
- The mapped 20–50 value is a simulated setpoint parameter, not °C.
- The Random Forest is evaluated using its original features and same synthetic
  response-generation pipeline. We have not trained it on potentiometer or LM35
  data.
- synthetic_reference_error is generated by the Python plant simulator; it is
  not a measured sensor value.
- Do not call this real LM35 testing, F446RE testing or ML validation on hardware.
