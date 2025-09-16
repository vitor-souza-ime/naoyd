#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import qi
import time
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import json

# YOLO v12 (Ultralytics)
from ultralytics import YOLO

# Detectron2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# ---------------------------
# Configurações
# ---------------------------
INTERVAL_SECONDS = 5  # intervalo entre capturas
OUTPUT_DIR = "nao_benchmark_results"
CONFIDENCE_THRESHOLD = 0.5

# Escolha da versão do YOLO11 para benchmark
# yolo11n.pt - Nano (mais rápido, menor precisão)
# yolo11s.pt - Small (balanceado)  
# yolo11m.pt - Medium (balanceado velocidade/precisão)
# yolo11l.pt - Large (alta precisão)
# yolo11x.pt - Extra-Large (máxima precisão, mais lento)
YOLO_MODEL_VERSION = "yolo12m.pt"  # Recomendado para benchmark balanceado

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------------------
# Funções utilitárias do NAO (mantidas do código original)
# ---------------------------

def connect_to_nao(ip="172.15.1.29", port=9559):
    """Conecta ao robô NAO"""
    try:
        session = qi.Session()
        session.connect(f"tcp://{ip}:{port}")
        print(f"✅ Conectado ao NAO em {ip}:{port}")
        return session
    except Exception as e:
        print(f"❌ Erro ao conectar ao NAO: {e}")
        raise

def capture_image_from_nao(session):
    """Captura imagem da câmera do NAO com tratamento robusto de erros"""
    try:
        camera_service = session.service("ALVideoDevice")
        camera_id = 0  # Câmera frontal
        resolution = 2  # VGA (640x480)
        color_space = 11  # RGB
        fps = 5
        
        # Método mais robusto de subscrição
        try:
            # Primeiro tenta desinscrever qualquer cliente existente
            try:
                camera_service.unsubscribe("python_client")
            except:
                pass
            
            # Subscreve à câmera
            video_client = camera_service.subscribeCamera("python_client", camera_id, resolution, color_space, fps)
        except Exception as e:
            print(f"⚠️ Método subscribeCamera falhou: {e}")
            # Método alternativo
            try:
                video_client = camera_service.subscribe("python_client", resolution, color_space, fps)
            except Exception as e2:
                print(f"⚠️ Método subscribe também falhou: {e2}")
                raise Exception("Não foi possível subscrever à câmera")
        
        # Aguarda um pouco para estabilizar
        time.sleep(0.2)
        
        # Captura a imagem
        nao_image = camera_service.getImageRemote(video_client)
        
        if nao_image is None or len(nao_image) < 7:
            raise Exception("Dados de imagem inválidos")
        
        # Extrai dados da imagem
        width = nao_image[0]
        height = nao_image[1]
        channels = nao_image[2]
        image_data = nao_image[6]
        
        print(f"📷 Imagem capturada: {width}x{height}, {channels} canais")
        
        # Converte para array numpy
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        
        # Verifica se o tamanho está correto
        expected_size = width * height * channels
        if len(image_array) != expected_size:
            print(f"⚠️ Tamanho inesperado da imagem: {len(image_array)} vs {expected_size}")
            image_array = image_array[:expected_size]
        
        image_array = image_array.reshape((height, width, channels))
        
        # Converte para PIL Image
        image = Image.fromarray(image_array).convert("RGB")
        
        return image
        
    except Exception as e:
        print(f"❌ Erro na captura principal: {e}")
        print("🔄 Tentando método alternativo...")
        return capture_image_alternative(session)
    finally:
        # Limpa a subscrição
        try:
            if 'video_client' in locals():
                camera_service.unsubscribe(video_client)
        except:
            pass

def capture_image_alternative(session):
    """Método alternativo usando ALPhotoCapture"""
    try:
        print("📸 Usando método alternativo ALPhotoCapture...")
        photo_service = session.service("ALPhotoCapture")
        photo_service.setResolution(2)  # VGA
        photo_service.setPictureFormat("jpg")
        
        # Caminho temporário
        temp_path = "/tmp/nao_temp_image.jpg"
        
        # Remove arquivo anterior se existir
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Tira a foto
        photo_service.takePicture(temp_path)
        
        # Aguarda o arquivo ser criado
        time.sleep(0.5)
        
        if not os.path.exists(temp_path):
            raise Exception("Arquivo de imagem não foi criado")
        
        # Carrega a imagem
        image = Image.open(temp_path).convert("RGB")
        print("✅ Imagem capturada com método alternativo")
        
        return image
        
    except Exception as e:
        print(f"❌ Método alternativo também falhou: {e}")
        print("🔵 Usando imagem de fallback...")
        # Retorna uma imagem azul como fallback
        return Image.new('RGB', (640, 480), color='blue')

# ---------------------------
# Configuração do YOLO v12
# ---------------------------

def setup_yolo():
    """Configura o modelo YOLO v12"""
    print(f"🔥 Configurando YOLO v12 ({YOLO_MODEL_VERSION})...")
    
    try:
        # Carrega o modelo YOLO v12
        model = YOLO(YOLO_MODEL_VERSION)
        
        print(f"✅ YOLO v12 ({YOLO_MODEL_VERSION}) configurado com sucesso")
        return model
        
    except Exception as e:
        print(f"❌ Erro ao configurar YOLO: {e}")
        raise

def detect_objects_yolo(model, image):
    """Detecta objetos usando YOLO v12"""
    try:
        # Converte PIL Image para numpy array se necessário
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        print(f"🔍 YOLO processando imagem: {image_np.shape}")
        
        start_time = time.time()
        results = model(image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
        elapsed = time.time() - start_time
        
        objects = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extrai informações da detecção
                    conf = float(box.conf.cpu().numpy()[0])
                    cls = int(box.cls.cpu().numpy()[0])
                    xyxy = box.xyxy.cpu().numpy()[0].tolist()
                    
                    objects.append({
                        "class": model.names[cls],
                        "score": conf,
                        "box": xyxy
                    })
        
        print(f"⏱️ YOLO detecção concluída em {elapsed:.3f}s - {len(objects)} objetos encontrados")
        return objects, results, elapsed
        
    except Exception as e:
        print(f"❌ Erro na detecção YOLO: {e}")
        raise

def draw_yolo_results(image, results):
    """Desenha resultados do YOLO na imagem"""
    try:
        # Converte PIL para numpy se necessário
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
            
        # YOLO results já vem com método plot
        annotated_image = results[0].plot()
        
        return annotated_image
        
    except Exception as e:
        print(f"❌ Erro ao desenhar resultados YOLO: {e}")
        # Retorna imagem original em caso de erro
        if isinstance(image, Image.Image):
            return np.array(image)
        else:
            return image

# ---------------------------
# Configuração do Detectron2 (mantida do código original)
# ---------------------------

def setup_detectron2(threshold=CONFIDENCE_THRESHOLD):
    """Configura o modelo Detectron2"""
    print("🤖 Configurando Detectron2...")
    
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        # Configura dispositivo (CPU ou GPU)
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            print("🚀 Detectron2 usando GPU para inferência")
        else:
            cfg.MODEL.DEVICE = "cpu"
            print("💻 Detectron2 usando CPU para inferência")
        
        predictor = DefaultPredictor(cfg)
        class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        
        print(f"✅ Detectron2 configurado com {len(class_names)} classes")
        return predictor, class_names
        
    except Exception as e:
        print(f"❌ Erro ao configurar Detectron2: {e}")
        raise

def detect_objects_detectron2(predictor, class_names, image):
    """Detecta objetos na imagem usando Detectron2"""
    try:
        # Converte PIL Image para numpy array (RGB -> BGR para OpenCV)
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        print(f"🔍 Detectron2 processando imagem: {image_np.shape}")
        
        start_time = time.time()
        outputs = predictor(image_np)
        elapsed = time.time() - start_time
        
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        
        objects = []
        for i in range(len(classes)):
            objects.append({
                "class": class_names[classes[i]],
                "score": float(scores[i]),
                "box": boxes[i].tolist()
            })
        
        print(f"⏱️ Detectron2 detecção concluída em {elapsed:.3f}s - {len(objects)} objetos encontrados")
        return objects, outputs, elapsed
        
    except Exception as e:
        print(f"❌ Erro na detecção Detectron2: {e}")
        raise

def draw_detectron2_results(image, outputs):
    """Desenha resultados do Detectron2 na imagem"""
    try:
        # Converte PIL para numpy se necessário
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Detectron2 espera BGR, então convertemos RGB->BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        v = Visualizer(image_bgr, MetadataCatalog.get("coco_2017_train"), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        result_image = out.get_image()
        
        return result_image
        
    except Exception as e:
        print(f"❌ Erro ao desenhar resultados Detectron2: {e}")
        # Retorna imagem original em caso de erro
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            return image

# ---------------------------
# Funções de exibição e salvamento
# ---------------------------

def display_results_comparison(yolo_image, detectron2_image, iteration):
    """Exibe resultados lado a lado para comparação"""
    try:
        # Redimensiona imagens se necessário
        h1, w1 = yolo_image.shape[:2]
        h2, w2 = detectron2_image.shape[:2]
        
        # Calcula dimensões para concatenação
        max_height = max(h1, h2)
        
        # Redimensiona para mesma altura
        if h1 != max_height:
            ratio = max_height / h1
            yolo_image = cv2.resize(yolo_image, (int(w1 * ratio), max_height))
        
        if h2 != max_height:
            ratio = max_height / h2
            detectron2_image = cv2.resize(detectron2_image, (int(w2 * ratio), max_height))
        
        # Concatena horizontalmente
        comparison_image = np.hstack([yolo_image, detectron2_image])
        
        # Adiciona texto indicativo
        cv2.putText(comparison_image, f"YOLO v12 (Iteracao {iteration})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison_image, "Detectron2", 
                   (yolo_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Exibe a imagem de comparação
        try:
            cv2.imshow("NAO Benchmark: YOLO vs Detectron2", comparison_image)
            cv2.waitKey(1)
        except Exception as e:
            print(f"⚠️ Não foi possível exibir a imagem: {e}")
        
        return comparison_image
        
    except Exception as e:
        print(f"❌ Erro ao exibir comparação: {e}")
        return None

def print_benchmark_results(yolo_objects, yolo_time, detectron2_objects, detectron2_time, iteration):
    """Imprime resultados do benchmark no console"""
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS - Iteração {iteration}")
    print(f"{'='*60}")
    
    print(f"\n🔥 YOLO v12:")
    print(f"   ⏱️  Tempo de processamento: {yolo_time:.3f}s")
    print(f"   🎯 Objetos detectados: {len(yolo_objects)}")
    if len(yolo_objects) > 0:
        print(f"   🏆 Maior confiança: {max([obj['score'] for obj in yolo_objects]):.3f}")
        print(f"   📋 Top detecções:")
        for i, obj in enumerate(sorted(yolo_objects, key=lambda x: x['score'], reverse=True)[:3], 1):
            print(f"      {i}. {obj['class']} (confiança: {obj['score']:.3f})")
    
    print(f"\n🤖 Detectron2:")
    print(f"   ⏱️  Tempo de processamento: {detectron2_time:.3f}s")
    print(f"   🎯 Objetos detectados: {len(detectron2_objects)}")
    if len(detectron2_objects) > 0:
        print(f"   🏆 Maior confiança: {max([obj['score'] for obj in detectron2_objects]):.3f}")
        print(f"   📋 Top detecções:")
        for i, obj in enumerate(sorted(detectron2_objects, key=lambda x: x['score'], reverse=True)[:3], 1):
            print(f"      {i}. {obj['class']} (confiança: {obj['score']:.3f})")
    
    print(f"\n⚡ Comparação de Velocidade:")
    speed_diff = abs(yolo_time - detectron2_time)
    if yolo_time < detectron2_time:
        print(f"   🏃 YOLO é {speed_diff:.3f}s mais rápido ({((detectron2_time/yolo_time-1)*100):.1f}% mais rápido)")
    else:
        print(f"   🏃 Detectron2 é {speed_diff:.3f}s mais rápido ({((yolo_time/detectron2_time-1)*100):.1f}% mais rápido)")

def save_benchmark_results(yolo_image, detectron2_image, comparison_image, 
                         yolo_objects, detectron2_objects, yolo_time, detectron2_time, iteration):
    """Salva todos os resultados do benchmark"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar imagens
        yolo_filename = os.path.join(OUTPUT_DIR, f"yolo_{iteration}_{timestamp}.jpg")
        detectron2_filename = os.path.join(OUTPUT_DIR, f"detectron2_{iteration}_{timestamp}.jpg")
        comparison_filename = os.path.join(OUTPUT_DIR, f"comparison_{iteration}_{timestamp}.jpg")
        
        cv2.imwrite(yolo_filename, yolo_image)
        cv2.imwrite(detectron2_filename, detectron2_image)
        if comparison_image is not None:
            cv2.imwrite(comparison_filename, comparison_image)
        
        # Salvar dados JSON
        json_filename = os.path.join(OUTPUT_DIR, f"benchmark_{iteration}_{timestamp}.json")
        benchmark_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "yolo": {
                "processing_time": yolo_time,
                "objects_count": len(yolo_objects),
                "objects": yolo_objects
            },
            "detectron2": {
                "processing_time": detectron2_time,
                "objects_count": len(detectron2_objects),
                "objects": detectron2_objects
            },
            "comparison": {
                "speed_winner": "YOLO" if yolo_time < detectron2_time else "Detectron2",
                "speed_difference": abs(yolo_time - detectron2_time),
                "total_objects_yolo": len(yolo_objects),
                "total_objects_detectron2": len(detectron2_objects)
            }
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Arquivos salvos:")
        print(f"   📸 YOLO: {yolo_filename}")
        print(f"   📸 Detectron2: {detectron2_filename}")
        print(f"   📸 Comparação: {comparison_filename}")
        print(f"   📊 Dados: {json_filename}")
        
        return benchmark_data
        
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
        return None

# ---------------------------
# Main loop
# ---------------------------

def main():
    print("=" * 80)
    print("🔥 BENCHMARK: NAO + YOLO v12 vs Detectron2")
    print(f"🔥 YOLO Model: {YOLO_MODEL_VERSION}")
    print(f"🤖 Detectron2 Model: Mask R-CNN ResNet-50")
    print("=" * 80)
    
    # Conecta ao NAO
    try:
        session = connect_to_nao()
    except Exception as e:
        print(f"❌ Falha na conexão com o NAO: {e}")
        print("💡 Verifique se o NAO está ligado e conectado na rede")
        return
    
    # Configura os modelos
    try:
        yolo_model = setup_yolo()
        detectron2_predictor, detectron2_class_names = setup_detectron2()
    except Exception as e:
        print(f"❌ Falha na configuração dos modelos: {e}")
        return
    
    print(f"\n🚀 Iniciando benchmark (intervalo: {INTERVAL_SECONDS}s)")
    print(f"🎯 Limiar de confiança: {CONFIDENCE_THRESHOLD}")
    print("Pressione Ctrl+C para parar\n")
    
    iteration = 1
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while True:
            print(f"\n{'='*50}")
            print(f"📸 BENCHMARK - Iteração {iteration}")
            print(f"{'='*50}")
            
            try:
                # Captura imagem do NAO
                print("📷 Capturando imagem...")
                image = capture_image_from_nao(session)
                
                if image is None:
                    print("❌ Falha ao capturar imagem")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"❌ Muitos erros consecutivos ({consecutive_errors}). Encerrando...")
                        break
                    time.sleep(INTERVAL_SECONDS)
                    continue
                
                # Detecção com YOLO v12
                print("\n🔥 Executando detecção YOLO v12...")
                yolo_objects, yolo_results, yolo_time = detect_objects_yolo(yolo_model, image)
                yolo_image = draw_yolo_results(image, yolo_results)
                
                # Detecção com Detectron2
                print("\n🤖 Executando detecção Detectron2...")
                detectron2_objects, detectron2_outputs, detectron2_time = detect_objects_detectron2(
                    detectron2_predictor, detectron2_class_names, image)
                detectron2_image = draw_detectron2_results(image, detectron2_outputs)
                
                # Exibir resultados no console
                print_benchmark_results(yolo_objects, yolo_time, detectron2_objects, detectron2_time, iteration)
                
                # Exibir imagens de comparação
                comparison_image = display_results_comparison(yolo_image, detectron2_image, iteration)
                
                # Salvar resultados
                benchmark_data = save_benchmark_results(
                    yolo_image, detectron2_image, comparison_image,
                    yolo_objects, detectron2_objects, yolo_time, detectron2_time, iteration
                )
                
                # Reset contador de erros
                consecutive_errors = 0
                iteration += 1
                
                # Aguarda próxima iteração
                print(f"\n⏳ Aguardando {INTERVAL_SECONDS}s para próxima iteração...")
                time.sleep(INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ Erro na iteração {iteration}: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"❌ Muitos erros consecutivos ({consecutive_errors}). Encerrando...")
                    break
                time.sleep(INTERVAL_SECONDS / 2)
    
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro crítico no benchmark: {e}")
    finally:
        # Limpeza
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("🧹 Limpeza concluída")
        print("👋 Encerrando benchmark")

if __name__ == "__main__":
    main()
