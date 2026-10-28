import cv2
import numpy as np

class CollisionAvoidanceAnalyzer:
    """
    Detecta aproximação de drones combinando:
    1. Expansão da área da Bounding Box do YOLO
    2. Análise do Optical Flow (colisão frontal estacionária ou convergência ao centro)
    """
    
    def __init__(self, clusters=5, fps=30):
        self.fps = max(1.0, float(fps))
        
        # Dicionários de rastreamento por drone
        self.nextBoxID = 0
        self.trackedBoxes = {}
        self.smoothedExpansions = {}
        self.smoothedVectors = {}     # Suaviza a seta para não girar com ruído
        self.dangerStreaks = {}
        self.lastAreas = {}
        
        # Limiares de ajuste
        self.expansionThreshold = 0.015  # Crescimento mínimo da área da caixa (1.5% por frame)
        self.directionThreshold = 0.65   # Alinhamento mínimo quando há movimento lateral
        self.stationaryThreshold = 1.2   # Abaixo disso (px/frame), os vetores se anularam (vindo reto na lente)
        self.centerZoneRadius = 50.0     # Raio (px) considerado centro da câmera
        
        self.alertCounter = 0

    def _matchBoxes(self, boxes):
        """Associa as caixas atuais do YOLO com as do frame anterior"""
        currentMatched = {}
        usedPrevIDs = set()

        for box in boxes:
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            area = float(max(1.0, (x2 - x1) * (y2 - y1)))

            bestID = None
            minDist = 80.0

            for bID, data in self.trackedBoxes.items():
                if bID in usedPrevIDs:
                    continue
                dist = np.linalg.norm(center - data['center'])
                if dist < minDist:
                    minDist = dist
                    bestID = bID

            if bestID is None:
                bestID = self.nextBoxID
                self.nextBoxID += 1
                self.smoothedExpansions[bestID] = 0.0
                self.smoothedVectors[bestID] = np.zeros(2, dtype=np.float32)
                self.dangerStreaks[bestID] = 0
                prevArea = area
            else:
                usedPrevIDs.add(bestID)
                prevArea = self.trackedBoxes[bestID]['area']

            currentMatched[bestID] = {
                'box': box,
                'center': center,
                'area': area,
                'prevArea': prevArea,
                'missed': 0
            }

        for bID, data in self.trackedBoxes.items():
            if bID not in currentMatched and data['missed'] < 5:
                data['missed'] += 1
                currentMatched[bID] = data

        self.trackedBoxes = currentMatched
        return [bID for bID, data in currentMatched.items() if data['missed'] == 0]

    def _checkBoxExpansion(self, boxID):
        """Mede a expansão suavizada da Bounding Box do YOLO"""
        data = self.trackedBoxes[boxID]
        currArea = data['area']
        prevArea = data['prevArea']

        rawExpansion = (currArea - prevArea) / (prevArea + 1e-6)
        self.smoothedExpansions[boxID] = (self.smoothedExpansions[boxID] * 0.65) + (rawExpansion * 0.35)
        
        isExpanding = self.smoothedExpansions[boxID] > self.expansionThreshold
        return isExpanding, self.smoothedExpansions[boxID]

    def _checkFlowDirection(self, boxID, box, trackingData, cameraCenter):
        """
        Analisa se o drone está no centro da câmera OU se os pontos do Optical Flow
        estão caminhando na direção do centro da câmera
        """
        if trackingData is None or len(trackingData.get('positions', [])) == 0:
            return False, 0.0, None, None, "SEM FLOW"

        positions = trackingData['positions']
        velocitiesPxFrame = trackingData['velocities'] / self.fps

        x1, y1, x2, y2 = box
        padX = (x2 - x1) * 0.10
        padY = (y2 - y1) * 0.10

        inBoxMask = (
            (positions[:, 0] >= x1 + padX) & (positions[:, 0] <= x2 - padX) &
            (positions[:, 1] >= y1 + padY) & (positions[:, 1] <= y2 - padY)
        )

        ptsInBox = positions[inBoxMask]
        vecsInBox = velocitiesPxFrame[inBoxMask]

        if len(ptsInBox) < 2:
            inBoxMask = (
                (positions[:, 0] >= x1) & (positions[:, 0] <= x2) &
                (positions[:, 1] >= y1) & (positions[:, 1] <= y2)
            )
            ptsInBox = positions[inBoxMask]
            vecsInBox = velocitiesPxFrame[inBoxMask]

        if len(ptsInBox) < 2:
            return False, 0.0, None, None, "POUCOS PTS"

        droneCenter = np.median(ptsInBox, axis=0)
        rawMoveVec = np.median(vecsInBox, axis=0)

        # Suavização do vetor para estabilizar a seta
        self.smoothedVectors[boxID] = (self.smoothedVectors[boxID] * 0.70) + (rawMoveVec * 0.30)
        smoothVec = self.smoothedVectors[boxID]
        moveNorm = np.linalg.norm(smoothVec)

        vecToCameraCenter = cameraCenter - droneCenter
        distToCenter = np.linalg.norm(vecToCameraCenter)

        # CENTRO se o drone estiver fisicamente dentro do raio central da câmera
        if distToCenter <= self.centerZoneRadius:
            return True, 1.0, droneCenter, smoothVec, "CENTRO"

        # PARADO se estiver fora do centro e quase parado (< 0.25 px/frame)
        if moveNorm < 0.25:
            return False, 0.0, droneCenter, None, "PARADO"

        # Se estiver fora do centro e em movimento, exige que aponte para o centro da câmera
        dirMove = smoothVec / moveNorm
        dirToCenter = vecToCameraCenter / (distToCenter + 1e-6)
        alignment = float(np.dot(dirMove, dirToCenter))

        isAligned = alignment >= self.directionThreshold
        return isAligned, alignment, droneCenter, smoothVec, f"{alignment:.2f}"

    def analyze(self, tracking_data, tracks, frame_size):
        """Decide aproximação a partir dos tracks do Kalman e do fluxo por ponto.

        A expansão da área só entra quando status == "updated". Em "predicted"
        a taxa suavizada e o streak ficam como estavam. A direção do fluxo
        é avaliada em todo frame. Tracks que o Kalman removeu saem do estado.
        """
        width, height = frame_size
        camera_center = np.array([width / 2.0, height / 2.0])
        active_ids = set(tracks.keys())
        for track_id in list(self.smoothedExpansions):
            if track_id not in active_ids:
                self.smoothedExpansions.pop(track_id, None)
                self.smoothedVectors.pop(track_id, None)
                self.dangerStreaks.pop(track_id, None)
                self.lastAreas.pop(track_id, None)

        by_id = {}
        for track_id, track in tracks.items():
            box = track["box"]
            self._ensure_track(track_id, box)
            status = track.get("status", "updated")
            if status == "updated":
                is_expanding, _exp_rate = self._update_expansion(track_id, box)
            else:
                is_expanding = self.smoothedExpansions[track_id] > self.expansionThreshold

            is_aligned, _alignment, _flow_center, _smooth_vec, _dir_status = self._checkFlowDirection(
                track_id, box, tracking_data, camera_center
            )

            if status == "updated":
                if is_expanding and is_aligned:
                    self.dangerStreaks[track_id] += 1
                else:
                    self.dangerStreaks[track_id] = max(0, self.dangerStreaks[track_id] - 1)
                if self.dangerStreaks[track_id] >= 3:
                    self.alertCounter = 15
                    self.dangerStreaks[track_id] = 0

            by_id[track_id] = {"perigo": bool(is_expanding and is_aligned)}

        alert = self.alertCounter > 0
        if self.alertCounter > 0:
            self.alertCounter -= 1
        return {"alert": alert, "by_id": by_id}

    def _ensure_track(self, track_id, box):
        if track_id in self.smoothedExpansions:
            return
        x1, y1, x2, y2 = [float(v) for v in box]
        self.smoothedExpansions[track_id] = 0.0
        self.smoothedVectors[track_id] = np.zeros(2, dtype=np.float32)
        self.dangerStreaks[track_id] = 0
        self.lastAreas[track_id] = float(max(1.0, (x2 - x1) * (y2 - y1)))

    def _update_expansion(self, track_id, box):
        """Expansão suavizada contra a última área medida, não contra a caixa prevista."""
        x1, y1, x2, y2 = [float(v) for v in box]
        curr_area = float(max(1.0, (x2 - x1) * (y2 - y1)))
        prev_area = self.lastAreas.get(track_id, curr_area)
        raw_expansion = (curr_area - prev_area) / (prev_area + 1e-6)
        self.smoothedExpansions[track_id] = (self.smoothedExpansions[track_id] * 0.65) + (raw_expansion * 0.35)
        self.lastAreas[track_id] = curr_area
        is_expanding = self.smoothedExpansions[track_id] > self.expansionThreshold
        return is_expanding, self.smoothedExpansions[track_id]

    def analyzeAndDraw(self, frame, trackingData, boxes=None):
        """Cruza a expansão da Bounding Box com a direção do Optical Flow e desenha o HUD"""
        resultFrame = frame.copy()
        h, w = resultFrame.shape[:2]
        cameraCenter = np.array([w / 2.0, h / 2.0])

        # Mira no centro da câmera
        cv2.drawMarker(resultFrame, (int(cameraCenter[0]), int(cameraCenter[1])),
                       (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Pontos do Optical Flow
        if trackingData is not None and len(trackingData.get('positions', [])) > 0:
            for pt in trackingData['positions']:
                cv2.circle(resultFrame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)

        if boxes is None or len(boxes) == 0:
            if self.alertCounter > 0:
                self._drawAlert(resultFrame)
                self.alertCounter -= 1
            return resultFrame

        activeBoxIDs = self._matchBoxes(boxes)

        for bID in activeBoxIDs:
            box = self.trackedBoxes[bID]['box']
            x1, y1, x2, y2 = map(int, box)

            # Expansão da Bounding Box
            isExpanding, expRate = self._checkBoxExpansion(bID)

            # Direção do Optical Flow
            isAligned, alignment, flowCenter, smoothVec, dirStatus = self._checkFlowDirection(
                bID, box, trackingData, cameraCenter
            )

            # Lógica de perigo
            if isExpanding and isAligned:
                self.dangerStreaks[bID] += 1
            else:
                self.dangerStreaks[bID] = max(0, self.dangerStreaks[bID] - 1)

            if self.dangerStreaks[bID] >= 3:
                self.alertCounter = 15
                self.dangerStreaks[bID] = 0

            # Desenha a seta sempre que houver um vetor de movimento válido fora do centro
            if flowCenter is not None and smoothVec is not None and dirStatus != "CENTRO":
                norm = np.linalg.norm(smoothVec)
                if norm >= 0.25:
                    dirVec = smoothVec / norm
                    pStart = (int(flowCenter[0]), int(flowCenter[1]))
                    pEnd = (int(flowCenter[0] + dirVec[0] * 30), int(flowCenter[1] + dirVec[1] * 30))
                    cv2.arrowedLine(resultFrame, pStart, pEnd, (255, 0, 255), 2, tipLength=0.3)

            # Cores: Vermelho (Perigo), Laranja (Expansão), Verde (Seguro)
            if isExpanding and isAligned:
                boxColor = (0, 0, 255)
            elif isExpanding:
                boxColor = (0, 165, 255)
            else:
                boxColor = (0, 255, 0)

            cv2.rectangle(resultFrame, (x1, y1), (x2, y2), boxColor, 2)
            label = f"ID:{bID} Exp:{expRate*100:.1f}% Dir:{dirStatus}"
            cv2.putText(resultFrame, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, boxColor, 1)

        if self.alertCounter > 0:
            self._drawAlert(resultFrame)
            self.alertCounter -= 1

        return resultFrame

    def _drawAlert(self, frame):
        cv2.putText(frame, "APROXIMACAO DETECTADA", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def cleanup(self):
        self.trackedBoxes.clear()
        self.smoothedExpansions.clear()
        self.smoothedVectors.clear()
        self.dangerStreaks.clear()
        self.lastAreas.clear()
        self.alertCounter = 0