import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

from functions import get_average_color, classify_bgr_color

# cap = cv2.VideoCapture("vid.mov")
cap = cv2.VideoCapture("demoRom.mp4")

ys = YOLOSegmentation("yolov8m-seg.pt")

font = cv2.FONT_HERSHEY_SIMPLEX

new_points = []

colors = []

player_coords = []

# INPUT TEAM COLORS (BGR)
team1_bgr = [255, 255, 255]
team2_bgr = [0, 0, 0]

# Lists to store the coordinates
og_perspective_coords = []
new_perspective_coords = []

# mapa que relacona coordenadas de la imagen original con las de la imagen top view
new_og_map = {}

# Callback function to get coordinates from the original image
def click_event_og(event, x, y, flags, params):
    global og_perspective_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(og_perspective_coords) < 4:
            og_perspective_coords.append((x, y))
            print(f"Original Image Coordinates: {og_perspective_coords}")
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("First Frame", frame)

# Callback function to get coordinates from the top-view image
def click_event_new(event, x, y, flags, params):
    global new_perspective_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(new_perspective_coords) < 4:
            new_perspective_coords.append((x, y))
            print(f"Top View Image Coordinates: {new_perspective_coords}")
            cv2.circle(dst, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Top View", dst)

def map_point(p, H):
    """Map a point using the homography matrix H."""
    point = np.array([p[0], p[1], 1.0])
    mapped_point = np.dot(H, point)
    return mapped_point[0] / mapped_point[2], mapped_point[1] / mapped_point[2]

new_ball_coords = [0, 0]
new_points_group1 = []
new_points_group2 = []

# Perspective transform function (pass in a point) (returns a point)
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

def is_offside(player_position, team1_positions, team2_positions, ball_position):
    if player_position in team1_positions:
        opposing_team_positions = team2_positions
    elif player_position in team2_positions:
        opposing_team_positions = team1_positions
    else:
        return False

    # verifico si hay jugadores del equipo contrario más cerca de la portería que el jugador actual
    if len(opposing_team_positions) >= 1:
        opposing_team_positions.sort(key=lambda pos: pos[0])
        last_player = opposing_team_positions[0] #defensa más cercano a la portería

        if player_position[0] < ball_position[0] and player_position[0] < last_player[0]:
            return True

    return False

# Read the first frame to get perspective coordinates
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

dst = cv2.imread("dst.png")

cv2.imshow("First Frame", frame)
cv2.imshow("Top View", dst)
cv2.setMouseCallback("First Frame", click_event_og)
cv2.setMouseCallback("Top View", click_event_new)

print("Please click 4 points on the original frame and the corresponding 4 points on the top-view image.")

# Wait until 4 points are selected for both original and new perspective coordinates
while len(og_perspective_coords) < 4 or len(new_perspective_coords) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()

# Loop through each frame
while True:
    # Video frame = frame
    ret, frame = cap.read()

    # 2D image = dst
    dst = cv2.imread("dst.png")

    if not ret:
        break

    # Copy of frame
    frame2 = np.array(frame)

    # Detect objects
    bboxes, classes, segmentations, scores = ys.detect(frame)
    # print("classes: ", classes)

    new_og_map.clear()
    player_coords.clear()
    colors.clear()
    new_points.clear()
    new_points_group1.clear()

    team1_positions = []
    team2_positions = []
    ball_position = None

    cont = 0
    
    # Loop through each object
    for index, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
        cont += 1
        # print("cont: ", cont)
        # If object is a player or the ball
        if class_id == 0 or class_id == 32:
            # 1. Dibujar cuadradito a cada jugador

            # Set corner coordinates for bounding box around player
            (x, y, x2, y2) = bbox
            
            # Draw segmentation around player
            if len(seg) != 0:
                minX = min(seg, key=itemgetter(0))[0]
                maxX = max(seg, key=itemgetter(0))[0]
                maxY = max(seg, key=itemgetter(1))[1]

                # Create smaller rectangle around player to use for color detection
                distLeft = int(abs(seg[0][0] - minX))
                distRight = int(abs(seg[0][0] - maxX))

                # Get smaller box points around player for detecting color
                newX = int((x2 - x)/3 + x)
                newY = int((y2 - y)/5 + y)
                newX2 = int(2*(x2 - x)/3 + x)
                newY2 = int(2*(y2 - y)/5 + y)

                if(class_id == 0):
                    # Shift color detection box based on player orientation
                    if(distRight > distLeft):
                        # Shift left
                        if(distLeft!=0):
                            newX = int(newX - ((distRight)/distLeft)/1.5)
                            newX2 = int(newX2 - ((distRight)/distLeft)/1.5)
                        else: 
                            newX = int(newX - 1)
                            newX2 = int(newX2 - 1)
                    else:
                        if(distRight!=0):
                            # Shift right
                            newX = int(newX + ((distLeft)/distRight)*1.5)
                            newX2 = int(newX2 + ((distLeft)/distRight)*1.5)
                        else:
                            newX = int(newX + 1)
                            newX2 = int(newX2 + 1)

                    # Define smaller rectangle around player to use for color detection
                    roi = frame2[newY:newY2, newX:newX2]

                    # Get average color of smaller rectangle
                    dominant_color = get_average_color(roi)
                    cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)
                    
                    # 2. Classify color as team1 or team2
                    team = classify_bgr_color(dominant_color, team1_bgr, team2_bgr)
                    # print("team:",team)


                    """
                    Aquí dibujar las líneas y polígonos en el video (frame) con coordenadas x, y-5 (para los jugadores y la pelota)
                    """
                    if(team == "group1"):
                        cv2.putText(frame, "Team 1", (x, y-5), font, 1, team1_bgr, 3, cv2.LINE_AA)
                        team1_positions.append((x, y-5))
                        
                        # Draw segmentation with the color of the dominant color of the player
                        cv2.polylines(frame, [seg], True, team1_bgr, 3)
                        cv2.circle(frame,(minX, maxY),5,team1_bgr,-1)
                    elif(team == "group2"):
                        cv2.putText(frame, "Team 2", (x, y-5), font, 1, team2_bgr, 3, cv2.LINE_AA)
                        team2_positions.append((x, y-5))

                        # Draw segmentation with the color of the dominant color of the player
                        cv2.polylines(frame, [seg], True, team2_bgr, 3)
                        cv2.circle(frame,(minX, maxY),5,team2_bgr,-1)
                elif(class_id == 32):
                    # Define smaller rectangle around ball to use for color detection
                    roi = frame2[newY:newY2, newX:newX2]

                    # Get average color of smaller rectangle
                    dominant_color = get_average_color(roi)
                    cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)

                    # 2. Classify color as ball (usar x,y permite dibujar en la posición correcta. NO USAR MINX, MINY)
                    team = "ball"
                    cv2.putText(frame, "Ball", (x, y-5), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
                    ball_position = (x, y-5)

                    # cv2.line(frame, (x, 0) , (x, 1690), (0, 0, 255), 2)

                    # # Dibujo una línea inclinada
                    # angle = 81 
                    # draw_inclined_line(frame, x, y, (0, 0, 255), 2, angle)
                    
                    # Draw segmentation with the color of the dominant color of the ball
                    cv2.polylines(frame, [seg], True, (0, 255, 255), 3)
                    cv2.circle(frame,(minX, maxY),5,(0, 255, 255),-1)

        # Perspective transform for each player
        if len(og_perspective_coords) == 4 and len(new_perspective_coords) == 4:
            perspective_transform([x, y-5], team, og_perspective_coords, new_perspective_coords)

    if ball_position is not None: # solo cuando la pelota se detecta porque sino da error en l
        for player_position in new_points_group1:
            if is_offside(player_position, new_points_group1, new_points_group2, new_ball_coords):
                cv2.putText(frame, "Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-12), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Not Offside", (new_og_map[player_position][0], new_og_map[player_position][1]-12), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    

    """
    Aquí dibujar las líneas y polígonos en la imagen top view (dst) con coordenadas max_point_X, max_point_Y (para los jugadores), y ball_coords[0], ball_coords[1] (para la pelota)
    """
   
    if new_points:
        print("new_points: ", new_points)
        print("new_points_group1: ", new_points_group1)
        print("new_points_group2: ", new_points_group2)
        # max_point_X, max_point_Y = min(new_points, key=itemgetter(0))[0], min(new_points, key=itemgetter(0))[1]
        if(new_points_group1): # si hay jugadores del equipo 1 detectados halla el que esté más cerca al arco contrario
            max_point_X, max_point_Y = min(new_points_group1, key=itemgetter(0))[0], min(new_points_group1, key=itemgetter(0))[1]
            cv2.circle(dst, (max_point_X, max_point_Y), 10, (0,255,255), 2)
            cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (0,255,255), 2)

    # Dibujar línea vertical para la pelota
    cv2.line(dst, (new_ball_coords[0], 0), (new_ball_coords[0], 1035), (0,0,255), 2)

    # Show images
    cv2.imshow("Img", frame)
    cv2.imshow("Top View", dst)

    # Space to move forward a frame
    t = 5
    key = cv2.waitKey(t) & 0xFF
    # Esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
