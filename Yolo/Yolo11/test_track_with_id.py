

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import random #
import time 


video_path = "Raw_Videos/fev_corte_3.mp4"  # Ajuste para o caminho do seu vídeo
model = YOLO("Weights\best_fev_2025.pt")  # Ajuste para o caminho do seu modelo
PROCESS_SCALE = 0.9  # Fator de redução de resolução para detecção
MAX_TRAJECTORY_LENGTH = 90  # Número máximo de pontos na trajetória


YOLO_CONFIDENCE_THRESHOLD = 0.6 # Minimum confidence for YOLO detection (Ajuste fino pode reduzir FPs)
IOU_THRESHOLD = 0.2 # Minimum IoU to associate a detection with an existing track (Ajuste fino pode reduzir FPs)
DETECTION_INTERVAL = 30 # Run YOLO detection every N frames
MIN_TRACK_AGE = 5 # Minimum number of frames a track must exist to be considered stable/displayed
NEW_TRACK_IOU_THRESHOLD = 0.3 # Stricter IoU threshold to prevent creating new tracks that overlap existing ones

def calculate_iou(boxA, boxB):
    """
    Calcula a Intersection over Union (IoU) entre dois bounding boxes.
    Boxes estão no formato (x, y, w, h).
    """
    
    x1A, y1A, wA, hA = boxA
    x2A, y2A = x1A + wA, y1A + hA
    x1B, y1B, wB, hB = boxB
    x2B, y2B = x1B + wB, y1B + hB

    
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    # Calcula a área da interseção
    interArea = max(0, xB - xA) * max(0, yB - yA)

    
    boxAArea = wA * hA
    boxBArea = wB * hB

    
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    iou = interArea / denominator
    return iou

def initialize_tracker(frame, bbox):
    """Inicializa e retorna um tracker CSRT com o bounding box fornecido."""
    # Garante que o bbox está dentro dos limites do frame
    h, w = frame.shape[:2]
    x, y, wb, hb = bbox
    x = max(0, x)
    y = max(0, y)
    wb = min(w - x, wb) # Garante que x+wb não exceda a largura
    hb = min(h - y, hb) # Garante que y+hb não exceda a altura

    if wb <= 0 or hb <= 0: # Caixa inválida
         print(f"Aviso: Tentativa de inicializar tracker com bbox inválido ou fora da tela: {(x, y, wb, hb)}")
         return None

    tracker = cv2.TrackerCSRT_create()
    try:
        # Usa uma cópia da ROI para inicialização, pode ser mais robusto
        roi = frame[y:y+hb, x:x+wb]
        if roi.size == 0:
             print(f"Aviso: ROI vazia para inicialização do tracker com bbox: {(x, y, wb, hb)}")
             return None
        tracker.init(frame, (x, y, wb, hb)) # Usa o frame completo mas a bbox ajustada
        return tracker
    except Exception as e:
        print(f"Erro ao inicializar o tracker: {e} com bbox {(x, y, wb, hb)}")
        return None


def main():
    tentativa = 1  # Identificador para o nome do vídeo de saída
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Propriedades do vídeo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # Configuração do VideoWriter para salvar o vídeo processado
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"teste{tentativa}_multi_track_fps_filtered.mp4"
    # Usa o FPS original do vídeo para a gravação
    out = cv2.VideoWriter(output_video_path, fourcc, fps_video, (frame_width, frame_height))

    # Criar uma janela para exibição
    window_name = "YOLO + CSRT Multi-Tracking (FPS + Filtered)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Tamanho inicial da janela

    trackers = {}      # {object_id: tracker}
    trajectories = {}  # {object_id: deque}
    colors = {}        # {object_id: (B, G, R)}
    track_start_frames = {} # {object_id: frame_count when started}
    next_object_id = 0
    frame_count = 0
    start_time_global = time.time() # Para FPS médio geral, se necessário

    while cap.isOpened():
        start_time_frame = time.time() # Início da medição de tempo para este frame
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro na leitura.")
            break

        frame_count += 1
        processed_frame = frame.copy() # Trabalha em uma cópia para desenhar

        # --- Atualização dos Trackers Existentes ---
        active_ids = list(trackers.keys())
        current_boxes = {} # Armazena as posições atualizadas {object_id: bbox}
        failed_ids = []

        for object_id in active_ids:
            tracker = trackers[object_id]
            success, box = tracker.update(processed_frame)

            if success:
                # Verifica se a caixa retornada é válida (não muito pequena ou fora da tela)
                x, y, w, h = [int(v) for v in box]
                if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame_width and (y + h) <= frame_height:
                    current_boxes[object_id] = box
                    # Atualiza trajetória
                    center_x = x + w // 2
                    center_y = y + h // 2
                    trajectories[object_id].append((center_x, center_y))
                else:
                    print(f"Track ID {object_id} produziu bbox inválido: {box}. Marcando como falha.")
                    failed_ids.append(object_id)
            else:
                # Marca para remoção se o rastreamento falhar
                failed_ids.append(object_id)
                # print(f"Track perdido para ID {object_id}") # Opcional: Descomentar para debug

        # Remove trackers que falharam
        for object_id in failed_ids:
            if object_id in trackers: del trackers[object_id]
            if object_id in trajectories: del trajectories[object_id]
            if object_id in colors: del colors[object_id] # Remove cor também
            if object_id in track_start_frames: del track_start_frames[object_id]


        # --- Detecção YOLO periódica ---
        if frame_count % DETECTION_INTERVAL == 0:
            # Redimensiona o frame para processamento com resolução reduzida
            scaled_frame = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            results = model(scaled_frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD) # Passa conf direto

            detected_boxes_scaled = results[0].boxes.xyxy.cpu().numpy()
            # confidences = results[0].boxes.conf.cpu().numpy() # Conf já filtrada pelo model()

            yolo_detections = [] # Lista de bboxes no formato (x, y, w, h) na escala original
            if detected_boxes_scaled.shape[0] > 0:
                for box_scaled in detected_boxes_scaled:
                    x1_s, y1_s, x2_s, y2_s = box_scaled[:4]
                    # Ajusta as coordenadas para a resolução original
                    x1 = int(x1_s / PROCESS_SCALE)
                    y1 = int(y1_s / PROCESS_SCALE)
                    x2 = int(x2_s / PROCESS_SCALE)
                    y2 = int(y2_s / PROCESS_SCALE)
                    w = x2 - x1
                    h = y2 - y1
                    # Filtro adicional: ignora caixas muito pequenas ou inválidas
                    if w > 5 and h > 5:
                        bbox_original = (x1, y1, w, h)
                        yolo_detections.append(bbox_original)


            # --- Associação de Detecções com Trackers Ativos ---
            tracked_ids_this_frame = list(current_boxes.keys()) # IDs que foram atualizados com sucesso
            used_detections = [False] * len(yolo_detections)
            matched_ids = set()

            if len(yolo_detections) > 0 and len(tracked_ids_this_frame) > 0:
                iou_matrix = np.zeros((len(tracked_ids_this_frame), len(yolo_detections)))
                for i, object_id in enumerate(tracked_ids_this_frame):
                    # Usa a caixa atualizada pelo tracker para calcular IoU
                    if object_id in current_boxes:
                        tracked_box = current_boxes[object_id]
                        for j, det_box in enumerate(yolo_detections):
                            iou_matrix[i, j] = calculate_iou(tracked_box, det_box)

                # Estratégia de associação (pode ser melhorada com Hungarian Algorithm)
                rows, cols = np.indices(iou_matrix.shape)
                
                # Itera pelas detecções, associando ao melhor tracker se IoU > threshold
                for j in range(len(yolo_detections)):
                    best_track_idx = -1
                    max_iou = IOU_THRESHOLD # Começa com o limiar mínimo
                    
                    # Encontra o melhor tracker para esta detecção
                    for i in range(len(tracked_ids_this_frame)):
                        if iou_matrix[i, j] > max_iou:
                             # Verifica se este tracker já não foi associado a uma detecção *melhor*
                             best_det_for_this_track = np.argmax(iou_matrix[i, :])
                             if best_det_for_this_track == j:
                                 max_iou = iou_matrix[i, j]
                                 best_track_idx = i

                    if best_track_idx != -1:
                        object_id = tracked_ids_this_frame[best_track_idx]
                        # Verifica se este tracker já não foi combinado com outra detecção (conflito)
                        if object_id not in matched_ids:
                            # Marca como usado
                            used_detections[j] = True
                            matched_ids.add(object_id)
                            # print(f"ID {object_id} associado com detecção {j} (IoU: {max_iou:.2f})")

                           


            # --- Inicializa Novos Trackers para Detecções Não Associadas ---
            for i, det_box in enumerate(yolo_detections):
                if not used_detections[i]:
                    # Verifica se a nova detecção não se sobrepõe significativamente com NENHUM tracker ativo
                    overlaps_existing = False
                    for tracked_id, tracked_box in current_boxes.items():
                         # Usa um limiar mais alto para evitar criar tracks muito próximos
                         if calculate_iou(det_box, tracked_box) > NEW_TRACK_IOU_THRESHOLD:
                             overlaps_existing = True
                             # print(f"Nova detecção {i} sobrepõe ID {tracked_id}, não criando novo tracker.")
                             break

                    if not overlaps_existing:
                        tracker = initialize_tracker(frame, det_box)
                        if tracker:
                            trackers[next_object_id] = tracker
                            trajectories[next_object_id] = deque(maxlen=MAX_TRAJECTORY_LENGTH)
                            track_start_frames[next_object_id] = frame_count # Registra frame de início
                            # Atualiza trajetória inicial
                            x, y, w, h = [int(v) for v in det_box]
                            center_x = x + w // 2
                            center_y = y + h // 2
                            trajectories[next_object_id].append((center_x, center_y))
                            # Gera cor aleatória
                            colors[next_object_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) # Evita cores muito escuras
                            
                            current_boxes[next_object_id] = det_box 
                            next_object_id += 1
                        

       
        valid_track_count = 0
        for object_id, box in current_boxes.items():
            # Verifica se o track é antigo o suficiente
            track_age = frame_count - track_start_frames.get(object_id, frame_count)
            if track_age >= MIN_TRACK_AGE:
                valid_track_count += 1
                x, y, w, h = [int(v) for v in box]
                color = colors.get(object_id, (0, 255, 0)) # Cor padrão verde

                # Desenha Bounding Box
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)

                # Desenha ID do objeto
                cv2.putText(processed_frame, f"ID: {object_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Desenha Trajetória (se houver pontos suficientes)
                if object_id in trajectories:
                    pts = list(trajectories[object_id])
                    # Desenha ponto central atual (último ponto da trajetória)
                    if pts:
                        center_x, center_y = pts[-1]
                        cv2.circle(processed_frame, (center_x, center_y), 4, (0, 0, 255), -1) # Ponto central em vermelho

                    # Desenha linhas da trajetória
                    for i in range(1, len(pts)):
                        if pts[i - 1] is None or pts[i] is None: continue
                        cv2.line(processed_frame, pts[i - 1], pts[i], color, 2)


        # --- Cálculo e Exibição de FPS ---
        end_time_frame = time.time()
        processing_time = end_time_frame - start_time_frame
        current_fps = 1.0 / processing_time if processing_time > 0 else 0

        # Adicionar texto informativo ao frame
        cv2.putText(processed_frame, f"FPS: {current_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Active Tracks: {valid_track_count}", (10, 60), # Mostra apenas tracks válidos (idade > min)
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mostrar o frame em uma janela
        cv2.imshow(window_name, processed_frame)

        # Escreve o frame processado no arquivo de saída
        out.write(processed_frame)

        # Verifica se o usuário pressionou a tecla 'q' para sair
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break



    print("Liberando recursos...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end_time_global = time.time()
    total_time = end_time_global - start_time_global
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processamento finalizado. Tempo total: {total_time:.2f}s, Média FPS: {avg_fps:.2f}")
    print(f"Vídeo salvo em: {output_video_path}")

if __name__ == '__main__':
    main()

