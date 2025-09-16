# 🤖 NAOYD - Benchmark YOLO v12 vs Detectron2 no Robô NAO

Este repositório contém um script em Python para realizar **benchmark de detecção de objetos** no robô humanoide **NAO**, comparando as arquiteturas **YOLO v12 (Ultralytics)** e **Detectron2 (Mask R-CNN ResNet-50)**.

## 📂 Estrutura
```

naoyd/
├── main.py        # Script principal de benchmark
├── requirements.txt (opcional)
└── README.md

````

## 🚀 Funcionalidades
- Conexão com o robô NAO via `qi.Session`.
- Captura de imagens da câmera do NAO com fallback seguro.
- Detecção de objetos com:
  - **YOLO v12 (Ultralytics)**
  - **Detectron2 (Mask R-CNN ResNet-50)**
- Comparação lado a lado dos resultados.
- Benchmark de:
  - Tempo de inferência
  - Quantidade de objetos detectados
  - Confiança máxima
- Salvamento automático de:
  - Imagens anotadas
  - Comparação YOLO vs Detectron2
  - Arquivos JSON com resultados

## ⚙️ Pré-requisitos

### Robô NAO
- NAO conectado à rede e acessível via IP.
- SDK **NAOqi** configurado na máquina host.

### Python
- Python **3.8+**
- Dependências principais:
  - `qi`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `torch`
  - `detectron2`
  - `ultralytics`

Instale as dependências com:

```bash
pip install -r requirements.txt
````

ou manualmente:

```bash
pip install opencv-python pillow numpy torch torchvision ultralytics
```

⚠️ Para instalar o **Detectron2**, consulte a [documentação oficial](https://github.com/facebookresearch/detectron2).

## ▶️ Uso

Clone o repositório:

```bash
git clone https://github.com/vitor-souza-ime/naoyd.git
cd naoyd
```

Edite o IP e porta do NAO no `main.py` se necessário:

```python
session = connect_to_nao(ip="172.15.1.29", port=9559)
```

Execute o script:

```bash
python main.py
```

## 📊 Resultados

A cada iteração, o benchmark salva em `nao_benchmark_results/`:

* Imagens anotadas pelo YOLO e Detectron2.
* Imagem de comparação lado a lado.
* Arquivo `.json` com os dados detalhados.

Exemplo de saída no console:

```
📊 BENCHMARK RESULTS - Iteração 1
🔥 YOLO v12:
   ⏱️ Tempo de processamento: 0.056s
   🎯 Objetos detectados: 3
   🏆 Maior confiança: 0.912
🤖 Detectron2:
   ⏱️ Tempo de processamento: 0.245s
   🎯 Objetos detectados: 2
   🏆 Maior confiança: 0.874
⚡ Comparação de Velocidade:
   🏃 YOLO é 0.189s mais rápido (336.6% mais rápido)
```

## 🧹 Limpeza

Ao encerrar (Ctrl+C), o script fecha as janelas abertas e libera recursos do NAO.

## 📌 Observações

* O modelo YOLO pode ser alterado editando a variável `YOLO_MODEL_VERSION` em `main.py`.
  Exemplos disponíveis:

  * `yolo12n.pt` (Nano, mais rápido)
  * `yolo12s.pt` (Small)
  * `yolo12m.pt` (Medium, recomendado)
  * `yolo12l.pt` (Large)
  * `yolo12x.pt` (Extra-Large, mais lento e preciso)

---

👤 Autor: [Vitor Souza](https://github.com/vitor-souza-ime)
📅 Projeto acadêmico para estudo comparativo de modelos de visão computacional aplicados em robôs humanoides.

