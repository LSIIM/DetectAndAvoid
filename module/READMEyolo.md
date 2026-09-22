# Ultralytics YOLO no NVIDIA Jetson — **Executar no JetPack 6.1**

Este README descreve uma instalação **nativa** (sem Docker) para executar o Ultralytics YOLO no **JetPack 6.1** (Ubuntu 22.04 / Python 3.10 em `aarch64`).

> Dica: use um ambiente virtual para evitar conflitos de sistema:
>
> ```bash
> python3 -m venv venv && source ./venv/bin/activate
> ```

---

## 1) Atualizar o sistema e instalar o Ultralytics

1. Atualize e instale o `pip` mais recente:

```bash
sudo apt update
sudo apt install -y python3-pip
pip install -U pip
```

2. Instale o pacote Ultralytics com dependências de exportação:

```bash
pip install "ultralytics[export]"
```

3. Reinicie o dispositivo:

```bash
sudo reboot
```

---

## 2) Instalar **PyTorch** e **Torchvision** para JP6.1
Os binários padrão do PyPI para `torch`/`torchvision` **não** são compatíveis com Jetson (ARM64). Instale as rodas (wheels) pré‑compiladas para **JetPack 6.1** e **Python 3.10**:

```bash
# PyTorch 2.5.0 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Torchvision 0.20 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

> **Compatibilidade**: consulte a página "PyTorch for Jetson" para outras combinações de JetPack/Python.

### 2.1) Corrigir dependência `cuSPARSELt` exigida pelo `torch 2.5`
Faça isso caso de algum problema relacionado a dependência
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

---

## 3) Instalar **onnxruntime-gpu** (aarch64)
O pacote do PyPI não fornece binários `aarch64` para Jetson. Use o wheel compatível com **JP6.1 + Python 3.10**:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

> **Ajuste de NumPy**: após instalar `onnxruntime-gpu`, fixe o NumPy conforme abaixo para evitar incompatibilidades:

```bash
pip install numpy==1.23.5
```

---

## 4) Uso da classe `kalman_filter` e do `track_id`

A classe `kalman_filter` foi pensada para acompanhar a mesma instância detectada ao longo do tempo. Ela armazena o estado do filtro por `track_id`, então o `track_id` precisa ser estável para o mesmo objeto entre frames consecutivos.

### 4.1) Como o `track_id` deve funcionar

O valor de `track_id` deve seguir estas regras:

- Deve ser único para cada objeto detectado no momento atual.
- Deve permanecer o mesmo para o mesmo objeto enquanto ele continuar visível.
- Não deve ser reutilizado para outro objeto antes que o anterior seja descartado ou perdido.
- Em geral, a melhor origem do `track_id` é a própria saída do YOLO (`results[0].boxes.id`), quando houver tracking interno habilitado.

Se o YOLO não entregar `boxes.id`, o código pode gerar IDs locais usando a lógica de associação dos boxe anteriores e da última detecção. Nesse caso, o fluxo esperado é:

1. Comparar detecção atual com a última detecção conhecida.
2. Usar IoU/distância para associar o objeto ao mesmo `track_id` anterior.
3. Se não houver correspondência, gerar um id novo (`_new_track_ids`).

### 4.2) Modo de uso recomendado

O uso correto do filtro é feito juntos com os IDs de detecção e as coordenadas da caixa:

```python
kalman = kalman_filter(process_noise=1e-2, measurement_noise=1e-1, prediction_horizon_sec=0.5)

boxes, confidences, ids, approach_detected = detector.process_frame(frame)
future_centers = kalman.process_kalman(ids, boxes, time.time())

for track_id in ids:
    predicted = kalman.predict_future_position(track_id, 0.5)
    if predicted is not None:
        print(track_id, predicted)
```

Nesse fluxo:

- `ids` representa os objetos ativos naquele frame.
- `boxes` contém as caixas detectadas.
- `process_kalman(...)` atualiza o estado do filtro com o centro de cada caixa e retorna as posições futuras estimadas.
- `predict_future_position(...)` usa o histórico do objeto para prever onde ele estará em um horizonte futuro.

### 4.3) O que acontece se o `track_id` estiver errado

Se dois objetos diferentes compartilharem o mesmo ID, ou se o mesmo objeto receber um ID diferente em frames sucessivos, o filtro de Kalman mistura estados distintos. O resultado costuma ser:

- previsão de posição instável;
- deslocamentos bruscos ou "saltos";
- associação incorreta entre objetos em frames seguintes;
- perda de precisão no cálculo de aproximação.

Em resumo: o filtro depende diretamente da continuidade do identificador. O `track_id` deve representar a mesma entidade ao longo do tempo.

### 4.4) Relação com o tracking do YOLO

No código do projeto, há dois cenários:

- `results[0].boxes.id is not None`: o YOLO já fornece o tracking, então o `track_id` deve ser reaproveitado diretamente.
- `results[0].boxes.id is None`: o código usa `_assign_track_ids(...)` para inferir associação entre detecções novas e antigas.

Em ambos os casos, a regra principal é a mesma: um objeto deve manter um único `track_id` enquanto continuar presente na cena.

### 4.5) Como usar `tracked_objects`

O parâmetro `tracked_objects` representa o estado do frame anterior. Ele deve conter, para cada objeto já identificado, o último box e o número de frames perdidos.

Estrutura esperada:

```python
tracked_objects = {
    12: {
        "box": [x1, y1, x2, y2],
        "frames_lost": 0,
    },
    13: {
        "box": [x1, y1, x2, y2],
        "frames_lost": 1,
    }
}
```

Esse dicionário é usado para:

- recuperar o último box do objeto;
- comparar a detecção atual com a última posição conhecida;
- decidir se o objeto continua sendo o mesmo;
- evitar que um ID novo seja gerado quando o mesmo objeto ainda está na cena.

Em outras palavras, `tracked_objects` é o histórico de associação da última iteração e funciona como a memória do tracker. Sem ele, o sistema perde a continuidade entre frames e o `track_id` vira inconsistente.

Exemplo de uso:

```python
boxes, confidences, ids, approach_detected = detector.process_frame(
    frame,
    tracked_objects=tracked_objects,
    box_offset=(0, 0),
    last_frame=previous_frame,
)

# Atualize o histórico para o próximo frame
tracked_objects = {
    track_id: {
        "box": box.tolist(),
        "frames_lost": 0,
    }
    for track_id, box in zip(ids, boxes)
}
```

> Observação: o `track_id` precisa ser consistente no `tracked_objects` do frame anterior para que a associação `ID atual ↔ ID anterior` funcione corretamente. Caso contrário, o filtro de Kalman e o tracker podem divergir e prever posições erradas.

---

## 5) Solução de problemas
- **Permissões do pip (Ubuntu 22.04)**: se instalar fora do `venv`, pode ser necessário `pip install --break-system-packages`.
- **Mismatch torch/torchvision**: valide se as versões acima correspondem ao seu JetPack e Python.
- **Para usar TensorRT**: no arquivo pyvenv.cfg, se desejar converter o modelo para TensorRT, adicione a seguinte linha:
```text
[sys]
include-system-site-packages = true 

```

---

# requirements-jetpack61.txt

Se preferir instalar via arquivo de requisitos, use o conteúdo abaixo e execute `pip install -r requirements-jetpack61.txt`:

```text
ultralytics[export]
# Wheels específicos para JP6.1 (Python 3.10, aarch64)
torch @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
torchvision @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
onnxruntime-gpu @ https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
numpy==1.23.5
```

---

**Referências**
- [Guia oficial Ultralytics — NVIDIA Jetson → *Run on JetPack 6.1*](https://docs.ultralytics.com/pt/guides/nvidia-jetson/#run-on-jetpack-61)


