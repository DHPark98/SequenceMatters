import cv2

# 입력 및 출력 파일 경로
input_file = "bicubic_chair.mp4"
output_file = "bicubic_chair_converted.avi"  # mp4 대신 avi 사용

# 비디오 캡처 및 저장 설정
cap = cv2.VideoCapture(input_file)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG 코덱 사용
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# 프레임 변환 및 저장
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
print(f"변환된 파일이 저장되었습니다: {output_file}")
