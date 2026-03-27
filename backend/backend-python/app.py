from flask import Flask, request, jsonify
import mysql.connector
from PIL import Image
import io
import open_clip
import torch
import numpy as np
import logging
from ultralytics import YOLO
from rembg import remove
import cv2

# Configuração
class Config:
    MYSQL_CONFIG = {
        'user': '',
        'password': '',
        'host': '',
        'database': ''
    }
    CLIP_MODEL = 'ViT-L-14'
    CLIP_PRETRAINED = 'openai'
    YOLO_MODEL = 'yolov8m-seg.pt'
    SIMILARITY_THRESHOLD = 0.88
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_config():
    return Config()

config = get_config()

# Configuração de logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(config)

# Carregar modelos
try:
    model_clip, _, preprocess = open_clip.create_model_and_transforms(config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip = model_clip.to(device)
    logger.info(f"Modelo CLIP carregado com sucesso no dispositivo: {device}")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo CLIP: {str(e)}")
    raise

try:
    model_yolo = YOLO(config.YOLO_MODEL)
    logger.info("Modelo YOLOv8 carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo YOLOv8: {str(e)}")
    raise

# Função para segmentar o animal na imagem com maior precisão
def segment_animal(image_bytes, confidence_threshold=0.35, iou_threshold=0.6):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        height, width, _ = image_np.shape

        # Ajustar confiança e IoU para maior precisão na segmentação
        yolo_results = model_yolo(image, conf=confidence_threshold, iou=iou_threshold)
        yolo_mask = None

        if yolo_results[0].masks and len(yolo_results[0].masks) > 0:
            # Pegar a máscara com maior área (assumindo o animal principal)
            max_area = 0
            best_mask = None
            for mask in yolo_results[0].masks.data.cpu().numpy():
                mask_bool = mask > 0.6  # Ajuste para um limiar mais alto na máscara
                area = np.sum(mask_bool)
                if area > max_area and area > 200:  # Aumentar o limite de área mínima
                    max_area = area
                    best_mask = mask_bool.astype(np.uint8)

            if best_mask is not None:
                yolo_mask = best_mask
                if yolo_mask.shape[:2] != (height, width):
                    yolo_mask = cv2.resize(yolo_mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # Verificar se a máscara cobre uma área significativa
                if np.sum(yolo_mask) < 200:  # Ajuste o limite de área mínima
                    logger.debug("Máscara YOLOv8 pequena demais, usando rembg")
                    yolo_mask = None
            else:
                logger.debug("Nenhuma máscara válida encontrada com YOLOv8, usando rembg")
                yolo_mask = None
        else:
            logger.debug("YOLOv8 não detectou máscaras, usando rembg")

        if yolo_mask is None or np.sum(yolo_mask) == 0:
            logger.debug("Usando rembg para segmentação")
            rembg_image = remove(image)
            rembg_np = np.array(rembg_image)
            if rembg_np.shape[:2] != (height, width):
                rembg_np = cv2.resize(rembg_np, (width, height), interpolation=cv2.INTER_NEAREST)
            # Pós-processamento no rembg para remover ruídos
            rembg_mask = (rembg_np[:, :, 3] > 0).astype(np.uint8)  # Usar o canal alfa
            rembg_mask_clean = cv2.morphologyEx(rembg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            final_image = image_np * np.expand_dims(rembg_mask_clean, axis=2)
            return Image.fromarray(final_image).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)

        # Pós-processamento avançado para remover ruídos e melhorar a máscara
        kernel = np.ones((5, 5), np.uint8)  # Kernel maior para maior suavização
        yolo_mask_clean = cv2.morphologyEx(yolo_mask, cv2.MORPH_OPEN, kernel)
        yolo_mask_clean = cv2.morphologyEx(yolo_mask_clean, cv2.MORPH_CLOSE, kernel)  # Fechamento para preencher buracos
        segmented_image = image_np * np.expand_dims(yolo_mask_clean, axis=2)
        segmented_image = Image.fromarray(segmented_image).convert('RGB')

        return segmented_image.resize((224, 224), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.error(f"Erro ao segmentar o animal com YOLOv8 e rembg: {str(e)}")
        return None

# Função para extrair características da imagem, focando no animal segmentado
def extract_features(image_bytes):
    try:
        segmented_image = segment_animal(image_bytes, confidence_threshold=0.35, iou_threshold=0.6)
        if segmented_image is None:
            logger.warning("Segmentação falhou, retornando None")
            return None
        # Normalizar a imagem segmentada para melhor extração de características
        image_preprocessed = preprocess(segmented_image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model_clip.encode_image(image_preprocessed)
        logger.debug(f"Características extraídas com sucesso para imagem segmentada de tamanho: {features.shape}")
        # Normalizar as características para melhorar a similaridade
        features_normalized = features.cpu().numpy().flatten() / np.linalg.norm(features.cpu().numpy().flatten())
        return features_normalized
    except Exception as e:
        logger.error(f"Erro ao extrair características da imagem: {str(e)}")
        return None

# Função para calcular similaridade entre duas características normalizadas
def calculate_similarity(features1, features2):
    try:
        if features1 is None or features2 is None:
            logger.warning("Uma das características é None, retornando similaridade 0")
            return 0.0
        similarity = np.dot(features1, features2)  # Usando características normalizadas, o coseno é direto
        logger.debug(f"Similaridade calculada (cosseno): {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"Erro ao calcular similaridade: {str(e)}")
        return 0.0

# Função para determinar a espécie principal detectada pela imagem
def detect_species(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        yolo_results = model_yolo(image, conf=0.35)
        if yolo_results[0].boxes and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                class_name = model_yolo.names[int(box.cls)]
                if 'dog' in class_name.lower() or 'cachorro' in class_name.lower():
                    return 'Cachorro'
                elif 'cat' in class_name.lower() or 'gato' in class_name.lower():
                    return 'Gato'
        return None  # Espécie não detectada
    except Exception as e:
        logger.error(f"Erro ao detectar espécie: {str(e)}")
        return None

# Endpoint de busca por imagem
@app.route('/search', methods=['POST'])
def search_similar_pets():
    logger.info("Recebendo requisição POST /search")
    if 'photo' not in request.files:
        logger.warning("Nenhuma imagem enviada na requisição")
        return jsonify({'results': []}), 200

    photo = request.files['photo']
    logger.info(f"Imagem recebida: {photo.filename}")
    try:
        # Verificar se é uma imagem válida
        if not photo.content_type.startswith('image/'):
            logger.warning(f"Arquivo enviado não é uma imagem: {photo.content_type}")
            return jsonify({'results': []}), 400

        uploaded_image_bytes = photo.read()
        logger.debug(f"Tamanho da imagem em bytes: {len(uploaded_image_bytes)}")

        # Detectar a espécie principal na imagem enviada
        species = detect_species(uploaded_image_bytes)
        if species is None:
            logger.warning("Espécie não detectada na imagem enviada")
            return jsonify({'results': []}), 200

        uploaded_features = extract_features(uploaded_image_bytes)
        if uploaded_features is None:
            logger.warning("Falha ao extrair características da imagem enviada")
            return jsonify({'results': []}), 200

        try:
            conn = mysql.connector.connect(**config.MYSQL_CONFIG)
            logger.info("Conectado ao banco MySQL com sucesso")
            cursor = conn.cursor()
            # Filtrar pets pela mesma espécie detectada
            cursor.execute("SELECT id, photo, species FROM pet WHERE status = 'lost' AND species = %s", (species,))
            pets = cursor.fetchall()
            logger.debug(f"Número de pets recuperados do banco (mesma espécie): {len(pets)}")

            similar_pets = []
            for pet_id, pet_photo, pet_species in pets:
                if pet_photo:
                    logger.debug(f"Processando pet com ID: {pet_id}, Espécie: {pet_species}")
                    pet_features = extract_features(pet_photo)
                    if pet_features is not None:
                        similarity = calculate_similarity(uploaded_features, pet_features)
                        if similarity > config.SIMILARITY_THRESHOLD:
                            similar_pets.append({'id': pet_id, 'similarity': float(similarity)})
                            logger.info(f"Pet similar encontrado - ID: {pet_id}, Similaridade: {similarity}, Espécie: {pet_species}")

            conn.close()
            logger.info("Conexão com o banco MySQL fechada")
            similar_pets.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"Total de pets similares encontrados: {len(similar_pets)}")
            return jsonify({'results': similar_pets})
        except mysql.connector.Error as mysql_err:
            logger.error(f"Erro ao conectar ou consultar o MySQL: {str(mysql_err)}")
            return jsonify({'results': []}), 200
    except Exception as e:
        logger.error(f"Erro ao processar a busca por imagem: {str(e)}")
        return jsonify({'results': []}), 200

if __name__ == '__main__':
    logger.info(f"Iniciando o servidor Flask em {config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
