import cv2
import numpy as np

class TrackedPoint:
    """Representa um ponto rastreado com um ID único"""
    _nextID = 0
    
    @classmethod
    def getNextID(cls):
        currentID = cls._nextID
        cls._nextID += 1
        return currentID
    
    @classmethod
    def resetCounter(cls):
        cls._nextID = 0
    
    def __init__(self, position):
        self.ID = TrackedPoint.getNextID()
        self.position = position

class OpticalFlowTracker:
    """
    Responsável estritamente pelo rastreio de pixels (Lucas-Kanade)
    Gera o ciclo de vida dos pontos e a poda matemática de anomalias (Drift)
    """
    
    def __init__(self, fps=30, processingSize=(640, 480)):
        self.fps = fps
        self.processingSize = processingSize
        self.maxPoint = 30
        self.maxPathLength = 20
        self.frameIter = 0
        
        self.feature_params = dict(
            maxCorners=self.maxPoint, 
            qualityLevel=0.01,
            minDistance=3,
            blockSize=3
        )
        
        self.lk_params = dict(
            winSize=(15, 15), 
            maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.oldGray = None
        self.p0 = None
        self.maskCanvas = np.zeros((processingSize[1], processingSize[0], 3), dtype=np.uint8)
        
        self.trackedPoints = {}
        self.pointPaths = {}

        self.yoloWasDetected = False

    def _initializeTracking(self, frameGray, detectionMask):

        self.oldGray = frameGray.copy()
        self.p0 = cv2.goodFeaturesToTrack(self.oldGray, mask=detectionMask, **self.feature_params)
        
        if self.p0 is None:
            return False
            
        self.maskCanvas = np.zeros_like(self.maskCanvas)
        self.trackedPoints.clear()
        self.pointPaths.clear()
        TrackedPoint.resetCounter()
        
        for point in self.p0.reshape(-1, 2):
            trackedPt = TrackedPoint(point)
            self.trackedPoints[trackedPt.ID] = trackedPt
            self.pointPaths[trackedPt.ID] = []
            
        return True

    def processFrame(self, frame, detectionMask=None):
        """
        Calcula o fluxo ótico e devolve um dicionário com os dados vetoriais limpos
        """
        if frame is None:
            return frame, None
            
        frameResized = cv2.resize(frame, self.processingSize)
        frameGray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)
        resultFrame = frameResized.copy()
        
        if self.p0 is None or self.oldGray is None or len(self.trackedPoints) < 3:
            self._initializeTracking(frameGray, detectionMask)
            return resultFrame, None
            
        pointIDs = list(self.trackedPoints.keys())
        self.p0 = np.array([self.trackedPoints[pID].position for pID in pointIDs]).reshape(-1, 1, 2)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.oldGray, frameGray, self.p0, None, **self.lk_params)
        
        goodIDs, goodNew, goodOld = [], [], []
        
        yoloDetected = False

        if detectionMask is not None:
            yoloDetected = cv2.countNonZero(detectionMask) > 0

        if yoloDetected and not self.yoloWasDetected:
            self.frameIter = self.fps
            
        self.yoloWasDetected = yoloDetected

        for i, (pID, status) in enumerate(zip(pointIDs, st.flatten())):

            if status == 1:
                x, y = int(p1[i][0][0]), int(p1[i][0][1])

                if yoloDetected:
                    if 0 <= x < self.processingSize[0] and 0 <= y < self.processingSize[1] and detectionMask[y, x] > 0:
                        goodIDs.append(pID); goodNew.append(p1[i][0]); goodOld.append(self.p0[i][0])
                else:
                    goodIDs.append(pID); goodNew.append(p1[i][0]); goodOld.append(self.p0[i][0])
                
        for pID in pointIDs:
            if pID not in goodIDs:
                self.trackedPoints.pop(pID, None)
                self.pointPaths.pop(pID, None)
                
        goodNew = np.array(goodNew)
        goodOld = np.array(goodOld)
        
        if len(goodNew) == 0:
            self._initializeTracking(frameGray, detectionMask)
            return resultFrame, None
            
        uvs = (goodNew - goodOld) * self.fps
        
        for i, pID in enumerate(goodIDs):
            self.trackedPoints[pID].position = goodNew[i]
            self.pointPaths[pID].append(goodNew[i])
            if len(self.pointPaths[pID]) > self.maxPathLength:
                self.pointPaths[pID] = self.pointPaths[pID][-self.maxPathLength:]

        validIDs, invalidIDs = self._checkConsistency(goodIDs, uvs)
        
        for pID in invalidIDs:
            self.trackedPoints.pop(pID, None)
            self.pointPaths.pop(pID, None)
            
        if len(validIDs) < len(goodIDs):
            validIndices = [i for i, pID in enumerate(goodIDs) if pID in validIDs]
            goodIDs = validIDs
            goodNew = goodNew[validIndices]
            uvs = uvs[validIndices]

        pathVectors = []

        for pID in goodIDs:
            path = self.pointPaths[pID]
            pathVectors.append(np.mean(path, axis=0) if len(path) > 0 else np.array([0.0, 0.0]))
            
        trackingData = {
            
            'IDs': goodIDs,
            'positions': goodNew,
            'velocities': uvs,
            'pathVectors': np.array(pathVectors)
        }

        # Reposição de pontos dentro da máscara do YOLO
        if self.frameIter >= int(0.5 * self.fps) - 1:

            if detectionMask is not None and cv2.countNonZero(detectionMask) > 0:

                newFeatures = cv2.goodFeaturesToTrack(self.oldGray, mask=detectionMask, **self.feature_params)

                if newFeatures is not None:

                    for newPt in newFeatures.reshape(-1, 2):

                        if not any(np.sum((extPt - newPt)**2) < 16 for extPt in goodNew):
                            tPt = TrackedPoint(newPt)
                            self.trackedPoints[tPt.ID] = tPt
                            self.pointPaths[tPt.ID] = []

            self.frameIter = -1
            
        self.frameIter += 1
        self.oldGray = frameGray.copy()
        
        self.maskCanvas = np.zeros_like(self.maskCanvas)

        for pID in goodIDs:
            path = self.pointPaths[pID]
            if len(path) > 1:
                cv2.polylines(self.maskCanvas, [np.array(path, dtype=np.int32).reshape((-1, 1, 2))], False, (100, 100, 100), 1)
                
        imgFinal = cv2.addWeighted(resultFrame, 1, self.maskCanvas, 0.5, 0)
        
        return imgFinal, trackingData

    def _checkConsistency(self, goodIDs, uvs):

        validIDs, invalidIDs = [], []

        for pID in goodIDs:

            isValID = True
            path = self.pointPaths[pID]

            if len(path) >= 5:
                recent = path[-5:]
                recentVels = [(recent[j+1] - recent[j]) * self.fps for j in range(len(recent)-1)]
                invalIDTrans, totalTrans = 0, 0

                for j in range(len(recentVels) - 1):
                    v1, v2 = recentVels[j], recentVels[j + 1]
                    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)

                    if mag1 > 2.0 and mag2 > 2.0:
                        totalTrans += 1
                        angleDiff = np.arccos(np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0))
                        
                        if angleDiff > np.pi / 2 or max(mag1, mag2) / (min(mag1, mag2) + 1e-6) > 3.0:
                            invalIDTrans += 1
                
                if totalTrans > 0 and (invalIDTrans / totalTrans) >= 0.85:
                    isValID = False
            
            if isValID: validIDs.append(pID)
            else: invalidIDs.append(pID)
        
        return validIDs, invalidIDs

    def cleanup(self):
        
        self.oldGray = None
        self.p0 = None
        self.trackedPoints.clear()
        self.pointPaths.clear()
        TrackedPoint.resetCounter()