import numpy as np

# 입력 파일명과 출력 파일명 지정
input_txt = "/home/cv/Desktop/yongjoon/Point-MAE/2048_nonorm/test_55/gt.txt"
output_ply = "/home/cv/Desktop/yongjoon/Point-MAE/2048_nonorm/test_55/gt.ply"

# RGB 값 지정 (예: 흰색)
default_rgb = (255, 255, 255)

# dense_points.txt 불러오기 (구분자 ; 또는 공백 자동 감지)
points = []
with open(input_txt, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # ; 또는 공백 구분 지원
        if ";" in line:
            vals = line.split(";")
        else:
            vals = line.split()
        xyz = [float(v) for v in vals[:3]]
        points.append(xyz)

points = np.array(points)
n_points = points.shape[0]

# PLY 파일 헤더 생성
header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

# PLY 파일로 저장
with open(output_ply, "w") as f:
    f.write(header)
    for xyz in points:
        line = "{:.8f} {:.8f} {:.8f} {} {} {}\n".format(
            xyz[0], xyz[1], xyz[2], *default_rgb
        )
        f.write(line)

print(f"변환 완료: {output_ply}")
