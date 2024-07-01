import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation
from functions import get_average_color, classify_bgr_color

cap = cv2.VideoCapture("messi_offsideTrap-cut.mp4")
ys = YOLOSegmentation("yolov8m-seg.pt")

font = cv2.FONT_HERSHEY_SIMPLEX

new_points = []
colors = []
player_coords = []

team1_bgr = [0, 0, 0]  # Placeholder, will be updated after color selection
team2_bgr = [0, 0, 0]  # Placeholder, will be updated after color selection

og_perspective_coords = []
new_perspective_coords = []
new_og_map = {}

selected_team1 = False
selected_team2 = False
goal_direction = None  # New variable to hold goal direction

def draw_inclined_line(image, x, y, color, thickness, angle_degrees):
    height, width = image.shape[:2]

    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Calculate the end points of the line using the angle
    length = max(height, width)
    x_end = int(x + length * np.cos(angle_radians))
    y_end = int(y - length * np.sin(angle_radians))

    # Draw the line
    cv2.line(image, (x, y), (x_end, y_end), color, thickness)

def click_event_color_selection(event, x, y, flags, params):
    global selected_team1, selected_team2, team1_bgr, team2_bgr
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = frame[y-10:y+10, x-10:x+10]
        dominant_color = get_average_color(roi)
        if not selected_team1:
            team1_bgr = dominant_color
            selected_team1 = True
            cv2.circle(frame, (x, y), 5, team1_bgr, -1)
            cv2.putText(frame, "Team 1 Selected", (x+20, y), font, 1, team1_bgr, 3, cv2.LINE_AA)
        elif not selected_team2:
            team2_bgr = dominant_color
            selected_team2 = True
            cv2.circle(frame, (x, y), 5, team2_bgr, -1)
            cv2.putText(frame, "Team 2 Selected", (x+20, y), font, 1, team2_bgr, 3, cv2.LINE_AA)
        cv2.imshow("Color Selection", frame)

def click_event_og(event, x, y, flags, params):
    global og_perspective_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(og_perspective_coords) < 4:
            og_perspective_coords.append((x, y))
            print(f"Original Image Coordinates: {og_perspective_coords}")
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("First Frame", frame)

def click_event_new(event, x, y, flags, params):
    global new_perspective_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(new_perspective_coords) < 4:
            new_perspective_coords.append((x, y))
            print(f"Top View Image Coordinates: {new_perspective_coords}")
            cv2.circle(dst, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Top View", dst)

def map_point(p, H):
    point = np.array([p[0], p[1], 1.0])
    mapped_point = np.dot(H, point)
    return mapped_point[0] / mapped_point[2], mapped_point[1] / mapped_point[2]

new_ball_coords = [0, 0]
new_points_group1 = []
new_points_group2 = []

def perspective_transform(player, team, original, new):
    src_points_np = np.array(original, dtype='float32')
    dst_points_np = np.array(new, dtype='float32')

    # Calculate the homography matrix
    H, status = cv2.findHomography(src_points_np, dst_points_np, cv2.RANSAC, 5.0)

    # Transform the player's point using the homography matrix
    player_np = np.array([player], dtype='float32').reshape(-1, 1, 2)
    new_pp = cv2.perspectiveTransform(player_np, H).reshape(-1, 2)[0]

    new_p = (int(new_pp[0]), int(new_pp[1]))

    # guardar el mapeo de las coordenadas de la imagen original con las de la imagen top view
    new_og_map[new_p] = player

    # Place transformed point for each player on dst
    if team == "group1":
        cv2.circle(dst, new_p, 10, team1_bgr, -1)
        new_points.append(new_p)
        new_points_group1.append(new_p)
    elif team == "group2":
        cv2.circle(dst, new_p, 10, team2_bgr, -1)
        new_points.append(new_p)
        new_points_group2.append(new_p)
    elif team == "ball":
        cv2.circle(dst, new_p, 10, (0, 255, 255), -1)
        new_ball_coords[0] = new_p[0]
        new_ball_coords[1] = new_p[1]

    cv2.imshow('Top View', dst)

def is_offside(player_position, team1_positions, team2_positions, goal_direction):
    if player_position in team1_positions:
        opposing_team_positions = team2_positions
    elif player_position in team2_positions:
        opposing_team_positions = team1_positions
    else:
        return False, -1
    if len(opposing_team_positions) >= 1:
        opposing_team_positions.sort(key=lambda pos: pos[0])
        last_player = opposing_team_positions[0] if goal_direction == 'left' else opposing_team_positions[-1]
        if goal_direction == 'left':
            if player_position[0] < last_player[0]:
                return True , round((last_player[0] - player_position[0])/15,2)
        elif goal_direction == 'right':
            if player_position[0] > last_player[0]:
                return True , round((player_position[0] - last_player[0])/15,2)
    return False, -1

def set_goal_direction(direction):
    global goal_direction
    goal_direction = direction
    cv2.destroyWindow("Goal Direction")
    cv2.destroyAllWindows()

def display_goal_direction_selection():
    global frame
    cv2.namedWindow("Goal Direction")
    cv2.putText(frame, "Is the goal on the left or right?", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press 'L' for Left or 'R' for Right", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Goal Direction", frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            set_goal_direction('left')
            break
        elif key == ord('r'):
            set_goal_direction('right')
            break

ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

cv2.namedWindow("Color Selection")
cv2.imshow("Color Selection", frame)
cv2.setMouseCallback("Color Selection", click_event_color_selection)

print("Por favor seleccione la camiseta de un jugador atacante. Luego, la de un jugador defensa. (en ese orden)")

while not selected_team1 or not selected_team2:
    cv2.waitKey(1)

cv2.destroyWindow("Color Selection")

dst = cv2.imread("dst3.png")

cv2.namedWindow("First Frame")
cv2.namedWindow("Top View")

cv2.imshow("First Frame", frame)
cv2.imshow("Top View", dst)

cv2.setMouseCallback("First Frame", click_event_og)
cv2.setMouseCallback("Top View", click_event_new)

print("Por favor haga clic en los vértices del cuadrilátero , de tal forma que un lado sea la línea del arco. Ambos deben coincidir lo más posible en ambas imágenes")

while len(og_perspective_coords) < 4 or len(new_perspective_coords) < 4:
    cv2.waitKey(1)

display_goal_direction_selection()

# Configurar el codec y crear el objeto VideoWriter para guardar el video procesado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video en formato mp4
out_3d = cv2.VideoWriter('output_3d.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))  # 20.0 es el frame rate
out_2d = cv2.VideoWriter('output_2d.mp4', fourcc, 20.0, (dst.shape[1], dst.shape[0]))  # 20.0 es el frame rate

numFrame = 0
# recorre todos los frames del video
while True:
    numFrame += 1
    ret, frame = cap.read()
    # print("frame:",frame)
    dst = cv2.imread("dst3.png")
    if not ret:
        break
    frame2 = np.array(frame)
    bboxes, classes, segmentations, scores = ys.detect(frame)
    new_og_map.clear()
    player_coords.clear()
    colors.clear()
    new_points.clear()
    new_points_group1.clear()
    new_points_group2.clear()
    team1_positions = []
    team2_positions = []
    ball_position = None

    # recorre todos los objetos hallados con YOLO
    for index, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
        if class_id == 0 or class_id == 32:
            (x, y, x2, y2) = bbox
            if len(seg) != 0:
                minX = min(seg, key=itemgetter(0))[0]
                maxX = max(seg, key=itemgetter(0))[0]
                maxY = max(seg, key=itemgetter(1))[1]
                distLeft = int(abs(seg[0][0] - minX))
                distRight = int(abs(seg[0][0] - maxX))
                # newX = int((x2 - x)/3 + x)
                # newY = int((y2 - y)/5 + y)
                # newX2 = int(2*(x2 - x)/3 + x)
                # newY2 = int(2*(y2 - y)/5 + y)

                newX = int((x2 - x) / 3 + x)
                newY = int(2 * (y2 - y) / 5 + y)  # Mover la ROI más abajo
                newX2 = int(2 * (x2 - x) / 3 + x)
                newY2 = int(3*(y2-y)/5 + y)  # Hasta el final del bounding box
                
                # Poner etiquetas a los objetos ("team1","team2" o "ball")
                if class_id == 0:
                    if distRight > distLeft:
                        if distLeft != 0:
                            newX = int(newX - ((distRight)/distLeft)/1.5)
                            newX2 = int(newX2 - ((distRight)/distLeft)/1.5)
                        else:
                            newX = int(newX - 1)
                            newX2 = int(newX2 - 1)
                    else:
                        if distRight != 0:
                            newX = int(newX + ((distLeft)/distRight)*1.5)
                            newX2 = int(newX2 + ((distLeft)/distRight)*1.5)
                        else:
                            newX = int(newX + 1)
                            newX2 = int(newX2 + 1)
                    roi = frame2[newY:newY2, newX:newX2]
                    dominant_color = get_average_color(roi)
                    cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)
                    team = classify_bgr_color(dominant_color, team1_bgr, team2_bgr)
                    if team == "group1":
                        #cv2.putText(frame, "Team 1", (x, y-5), font, 1, team1_bgr, 3, cv2.LINE_AA)
                        team1_positions.append((x, y-5))
                        cv2.polylines(frame, [seg], True, team1_bgr, 3)
                        cv2.circle(frame, (minX, maxY), 5, team1_bgr, -1)
                    elif team == "group2":
                        #cv2.putText(frame, "Team 2", (x, y-5), font, 1, team2_bgr, 3, cv2.LINE_AA)
                        team2_positions.append((x, y-5))
                        cv2.polylines(frame, [seg], True, team2_bgr, 3)
                        cv2.circle(frame, (minX, maxY), 5, team2_bgr, -1)
                elif class_id == 32:
                    roi = frame2[newY:newY2, newX:newX2]
                    dominant_color = get_average_color(roi)
                    cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)
                    team = "ball"
                    #cv2.putText(frame, "Ball", (x, y-5), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
                    ball_position = (x, y-5)
                    cv2.polylines(frame, [seg], True, (0, 255, 255), 3)
                    cv2.circle(frame, (minX, maxY), 5, (0, 255, 255), -1)
            
            # Hallar posición del jugador en campo de fútbol 2D
            if len(og_perspective_coords) == 4 and len(new_perspective_coords) == 4:
                perspective_transform([x, y-5], team, og_perspective_coords, new_perspective_coords)

    # Detectar si el jugador está en offside usando coordenadas 2D
    # for player_position in new_points_group1:
    #     offside, distance = is_offside(player_position, new_points_group1, new_points_group2, goal_direction)
    #     if offside:
    #         cv2.putText(frame, "Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-32), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    #         cv2.putText(frame, f"{distance} m", (new_og_map[player_position][0], new_og_map[player_position][1]-64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # else:
        #     cv2.putText(frame, "Not Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-32), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            

    # Trazar una recta amarilla al atacante más cercano al arco, otra recta azul al defensa más cercano a al arco y una recta roja a la pelota
    atacante_x = 0
    atacante_y = 0
    if new_points:
        if goal_direction == 'left':
            if new_points_group1:
                max_point_X, max_point_Y = min(new_points_group1, key=itemgetter(0))[0], min(new_points_group1, key=itemgetter(0))[1]
                atacante_x = max_point_X
                atacante_y = max_point_Y
                cv2.circle(dst, (max_point_X, max_point_Y), 10, (0, 255, 255), 2)
                cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (0, 255, 255), 2)
            if new_points_group2:
                max_point_X, max_point_Y = min(new_points_group2, key=itemgetter(0))[0], min(new_points_group2, key=itemgetter(0))[1]
                cv2.circle(dst, (max_point_X, max_point_Y), 10, (0, 255, 255), 2)
                cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (255, 0, 0), 2)
        else:
            if new_points_group1:
                max_point_X, max_point_Y = max(new_points_group1, key=itemgetter(0))[0], max(new_points_group1, key=itemgetter(0))[1]
                cv2.circle(dst, (max_point_X, max_point_Y), 10, (0, 255, 255), 2)
                cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (0, 255, 255), 2)
                atacante_x = max_point_X
                atacante_y = max_point_Y
            if new_points_group2:
                max_point_X, max_point_Y = max(new_points_group2, key=itemgetter(0))[0], max(new_points_group2, key=itemgetter(0))[1]
                cv2.circle(dst, (max_point_X, max_point_Y), 10, (0, 255, 255), 2)
                cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (255, 0, 0), 2)
    cv2.line(dst, (new_ball_coords[0], 0), (new_ball_coords[0], 1035), (0, 0, 255), 2)
    # Dibujar linea de offside en el frame 228 (cuando el jugador está en offside)

    if numFrame >= 60 and numFrame <= 64:
    # if numFrame>0:
        for player_position in new_points_group1:
            offside, distance = is_offside(player_position, new_points_group1, new_points_group2, goal_direction)
            if offside:
                #cv2.putText(frame, "Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-32), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                #cv2.putText(frame, f"{distance} m", (new_og_map[player_position][0], new_og_map[player_position][1]-64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # dibujar un rectángulo alrededor del jugador
                # cv2.rectangle(dst, (max_point_X-10, max_point_Y-10), (max_point_X+10, max_point_Y+10), (0, 255, 255), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # Get coordinates and size of the square
                x, y = new_og_map[(atacante_x, atacante_y)]
                w, h = 100,100

                # # Apply the mask
                # frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
                # frame[0:y, :] = gray[0:y, :]
                # frame[y+h:, :] = gray[y+h:, :]
                # frame[y:y+h, 0:x] = gray[y:y+h, 0:x]
                # frame[y:y+h, x+w:] = gray[y:y+h,x+w:]
                cv2.putText(frame, "Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-32), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{distance} m", (new_og_map[player_position][0], new_og_map[player_position][1]-64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        print("Dibujando linea de offside")
        cv2.line(frame, (418, 1077), (1902, 280), (0, 0, 255), 2)

    cv2.imshow("Img", frame)
    cv2.imshow("Top View", dst)
    t = 5
    key = cv2.waitKey(t) & 0xFF
    out_3d.write(frame)  # Guardar el frame procesado en el archivo de salida
    out_2d.write(dst)
    


    print(f"Frame {numFrame} procesado")

    if key == 27:
        break

cap.release()
out_3d.release()
out_2d.release()
cv2.destroyAllWindows()
