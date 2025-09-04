import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import mediapipe as mp
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

from model_8_keys import TinyTransformerModel

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.get_logger().info("Nó de reconhecimento de gestos iniciado.")

        # Obtenção dos parâmetros
        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_topic', '/camera/image_raw'),
                ('prediction_topic', '/gesture_prediction'),
                ('debug_image_topic', '/gesture_debug_image'),
                ('model.input_features', 16),
                ('model.d_model', 128),
                ('model.nhead', 4),
                ('model.num_encoder_layers', 3),
                ('model.dim_feedforward', 256),
                ('model.num_classes', 13),
                ('model.sequence_length', 60),
                ('model_path', 'models/model.pth'),
                ('class_names', ['default_class']),
                ('mediapipe.min_detection_confidence', 0.5),
                ('mediapipe.min_tracking_confidence', 0.5),
                ('mediapipe.joints_to_use', [0]),
                ('device', 'cpu')
            ])
        
        # Tópicos
        image_topic = self.get_parameter('image_topic').value
        prediction_topic = self.get_parameter('prediction_topic').value
        debug_image_topic = self.get_parameter('debug_image_topic').value

        # Parâmetros para previsão
        self.sequence_length = self.get_parameter('model.sequence_length').value
        self.class_names = self.get_parameter('class_names').value
        self.joints_to_use = self.get_parameter('mediapipe.joints_to_use').value
        self.device = self.get_parameter('device').value
        
        # MediaPipe
        min_detection_confidence = self.get_parameter('mediapipe.min_detection_confidence').value
        min_tracking_confidence = self.get_parameter('mediapipe.min_tracking_confidence').value
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

        # Carrega o modelo
        self.load_model()
        
        # Subs e Pubs
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.prediction_publisher = self.create_publisher(String, prediction_topic, 10)
        self.debug_image_publisher = self.create_publisher(Image, debug_image_topic, 10)

        # --- Variáveis de Estado ---
        self.sequence_data = []
        self.current_prediction = ""
        
        self.get_logger().info(f'Nó configurado. Escutando o tópico de imagem: "{image_topic}"')

    def load_model(self):
        """Carrega o modelo"""
        pkg_name = 'detector'
        package_path = get_package_share_directory(pkg_name)
        model_path = os.path.join(package_path, self.get_parameter('model_path').value)
        
        self.model = TinyTransformerModel(
            input_features=self.get_parameter('model.input_features').value,
            d_model=self.get_parameter('model.d_model').value,
            nhead=self.get_parameter('model.nhead').value,
            num_encoder_layers=self.get_parameter('model.num_encoder_layers').value,
            dim_feedforward=self.get_parameter('model.dim_feedforward').value,
            num_classes=self.get_parameter('model.num_classes').value
        )
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info(f"Modelo carregado com sucesso de: {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"Arquivo do modelo não encontrado em: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Erro ao carregar o modelo: {e}")

    def extract_and_process_keypoints(self, pose_landmarks):
        """Extrai e processa os keypoints a partir dos landmarks do MediaPipe."""
        try:
            if not pose_landmarks:
                return None

            # Calcula landmark do pescoço    
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            neck_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2

            neck_landmark = Landmark(x=neck_x, y=neck_y, visibility=neck_visibility)

            # Todos os landmarks
            all_landmarks_list = list(pose_landmarks.landmark)
            all_landmarks_list.append(neck_landmark) # Adiciona pescoço como landmark 33

            processed_keypoints = []
            for joint_index in self.joints_to_use:
                landmark = all_landmarks_list[joint_index]
                if hasattr(landmark, 'visibility') and landmark.visibility < 0.5:
                    return None 
                processed_keypoints.extend([landmark.x, landmark.y])
            
            return np.array(processed_keypoints)
        except Exception as e:
            self.get_logger().warn(f"Erro ao extrair keypoints: {e}")
            return None

    def image_callback(self, msg):
        """Callback principal que processa cada frame recebido."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Erro ao converter imagem: {e}")
            return

        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        display_text = ""
        keypoints = self.extract_and_process_keypoints(results.pose_landmarks)

        if keypoints is not None:
            display_text = self.current_prediction
            self.sequence_data.append(keypoints)

            if len(self.sequence_data) == self.sequence_length:
                self.predict()
                display_text = self.current_prediction
                self.sequence_data.clear() # Limpa para a próxima sequência
        else:
            self.sequence_data.clear()
            self.current_prediction = "Pose Incompleta"
            display_text = "Pose Incompleta"

        # Publica a predição em texto
        prediction_msg = String()
        prediction_msg.data = self.current_prediction if self.current_prediction else "N/A"
        self.prediction_publisher.publish(prediction_msg)

        # Prepara e publica a imagem de debug
        self.publish_debug_image(cv_image, results, display_text)

    def predict(self):
        """Realiza a previsão para 60 frames"""
        sequence_tensor = torch.tensor(np.array(self.sequence_data), dtype=torch.float32).to(self.device)
        sequence_tensor = sequence_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(sequence_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            self.current_prediction = self.class_names[predicted_idx]
            self.get_logger().info(f"Gesto previsto: {self.current_prediction}")

    def publish_debug_image(self, image, results, text):
        """Desenha informações na imagem e a publica."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_image_publisher.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Erro ao publicar imagem de debug: {e}")
    

if __name__ == '__main__':
    rclpy.init()
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()