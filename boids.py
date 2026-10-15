import pygame

import numpy as np

import time
from itertools import combinations

N_DRONES = 600
DRONE_SIZE = 5
DRONE_VELOCITY = 1
DISTANCE_TRESHOLD = 150
SEPARATION_STRENGTH = 15
ALIGNMENT_STRENGTH = .5
COHESION_STRENGTH = .2
VELOCITY_MAX_LIMIT = 10
VELOCITY_MIN_LIMIT = 5
STEERING_LIMIT = .8

X = 1280
Y = 1280

global drone_pos
global drone_directions

global text
global textRect
global screen

global neighbour_times
global averaging_times

neighbour_times = []
averaging_times = []

drone_pos = np.random.randint(
    [DRONE_SIZE, DRONE_SIZE], [X - DRONE_SIZE, Y - DRONE_SIZE], size=(N_DRONES, 2)
).astype(float)
drone_directions = np.random.choice([-1, 1], size=[N_DRONES, 2]).astype(float)
colors = np.random.randint(0, 255, size=(N_DRONES, 3))


def get_pos(drone_idx):
    return drone_pos[drone_idx]


def get_direction(drone_idx):
    return drone_directions[drone_idx]


def update_pos(drone_idx, x_direction=0, y_direction=0):
    drone_pos[drone_idx][0] += x_direction * DRONE_VELOCITY
    drone_pos[drone_idx][1] += y_direction * DRONE_VELOCITY


def update_location(drone_idx, x_loc=None, y_loc=None):
    if x_loc:
        drone_pos[drone_idx][0] += x_loc

    if y_loc:
        drone_pos[drone_idx][1] += y_loc


def update_direction(drone_idx, x_direction=None, y_direction=None):
    if x_direction is not None:
        drone_directions[drone_idx][0] = x_direction

    if y_direction is not None:
        drone_directions[drone_idx][1] = y_direction


def check_wall_collisions():
    # Check wall collisions
    for drone_idx in range(N_DRONES):
        if drone_pos[drone_idx][0] >= X - DRONE_SIZE:
            drone_pos[drone_idx][0] = X - DRONE_SIZE
            if drone_directions[drone_idx][0] > 0:
                drone_directions[drone_idx][0] *= -1

        elif drone_pos[drone_idx][0] <= DRONE_SIZE:
            drone_pos[drone_idx][0] = DRONE_SIZE
            if drone_directions[drone_idx][0] < 0:
                drone_directions[drone_idx][0] *= -1

        if drone_pos[drone_idx][1] >= Y - DRONE_SIZE:
            drone_pos[drone_idx][1] = Y - DRONE_SIZE
            if drone_directions[drone_idx][1] > 0:
                drone_directions[drone_idx][1] *= -1

        elif drone_pos[drone_idx][1] <= DRONE_SIZE:
            drone_pos[drone_idx][1] = DRONE_SIZE
            if drone_directions[drone_idx][1] < 0:
                drone_directions[drone_idx][1] *= -1

def collisions():
    check_wall_collisions()

    # Assign each drone to a grid cell
    drone_to_cell_assignment = {
        drone_idx:(np.floor(pos_x/(2*DRONE_SIZE)).astype(int), np.floor(pos_y/(2*DRONE_SIZE)).astype(int)) 
        for drone_idx, (pos_x, pos_y) in enumerate(drone_pos)
    }

    # Cells containing drones
    cell_to_drone_assignment = {}
    for drone_idx, cell in drone_to_cell_assignment.items():
        if cell in cell_to_drone_assignment:
            cell_to_drone_assignment[cell].append(drone_idx)
        else:
            cell_to_drone_assignment[cell] = [drone_idx]

    # Neighbour search
    candidate_pair_list = []
    for curr_cell, drones_in_curr_cell in cell_to_drone_assignment.items():
        for drone_idx in drones_in_curr_cell:
            cx = curr_cell[0]
            cy = curr_cell[1]

            neighbouring_cells = [
                (cx-1, cy-1), (cx-1, cy), (cx-1, cy+1), 
                (cx, cy-1),   (cx, cy),   (cx, cy+1), 
                (cx+1, cy-1), (cx+1, cy), (cx+1, cy+1)]

            for neighbour_cell in neighbouring_cells:
                for nb_drone in cell_to_drone_assignment.get(neighbour_cell, []):
                    if nb_drone > drone_idx:
                        candidate_pair_list.append((drone_idx, nb_drone))

    drone_indices = sorted(candidate_pair_list)

    for drone_idx_1, drone_idx_2 in drone_indices:
        d_x = drone_pos[drone_idx_1][0] - drone_pos[drone_idx_2][0]
        d_y = drone_pos[drone_idx_1][1] - drone_pos[drone_idx_2][1]

        distance = np.sqrt(d_x**2 + d_y**2)

        if distance <= (DRONE_SIZE * 2):
            # Find overlap depth
            overlap_depth = DRONE_SIZE * 2 - distance

            # Find overlap direction
            if distance > 0:
                overlap_d_x = d_x / distance
                overlap_d_y = d_y / distance

                x_correction = overlap_d_x * overlap_depth / 2
                y_correction = overlap_d_y * overlap_depth / 2
            else:
                overlap_d_x = 1
                overlap_d_y = 0

                x_correction = overlap_d_x * overlap_depth / 2
                y_correction = overlap_d_y * overlap_depth / 2

            update_location(drone_idx_1, x_correction, y_correction)
            update_location(drone_idx_2, -x_correction, -y_correction)

            # Calculate relative velocity
            drone_velocity_1 = drone_directions[drone_idx_1] * DRONE_VELOCITY
            drone_velocity_2 = drone_directions[drone_idx_2] * DRONE_VELOCITY

            relative_velocity = drone_velocity_1 - drone_velocity_2
            velocity_direction = np.dot(relative_velocity, [overlap_d_x, overlap_d_y])

            if velocity_direction < 0:
                normal = np.array([overlap_d_x, overlap_d_y])

                new_velocity_1 = drone_velocity_1 - velocity_direction * normal
                new_velocity_2 = drone_velocity_2 + velocity_direction * normal
    
                new_direction_1 = new_velocity_1 / DRONE_VELOCITY
                new_direction_2 = new_velocity_2 / DRONE_VELOCITY

                update_direction(
                    drone_idx=drone_idx_1,
                    x_direction=new_direction_1[0],
                    y_direction=new_direction_1[1],
                )
                update_direction(
                    drone_idx=drone_idx_2,
                    x_direction=new_direction_2[0],
                    y_direction=new_direction_2[1],
                )

    check_wall_collisions()


def draw_arrow(start, end, color):
    start_vec = pygame.Vector2(start)
    end_vec = pygame.Vector2(end)
    
    # 1. Draw the main shaft line
    pygame.draw.line(screen, color, start_vec, end_vec, 3)
    
    # 2. Calculate the vector of the line and its angle
    arrow_vec = end_vec - start_vec
    if arrow_vec.length() == 0:
        return  # Prevent errors if start and end are the same point
    
    # Normalize the vector to get the direction, then scale by head size
    arrow_dir = arrow_vec.normalize()
    
    # Calculate the two fin points of the arrow head
    # Rotate the direction vector by 150 and -150 degrees to sweep backwards
    fin1 = end_vec + arrow_dir.rotate(150) * 15
    fin2 = end_vec + arrow_dir.rotate(-150) * 15
    
    # 3. Draw the arrow head (as a filled triangle polygon)
    pygame.draw.polygon(screen, color, [end_vec, fin1, fin2])


def find_neighbours(drone_idx, pairwise_displacements, pairwise_squared_distances):
    displacements = pairwise_displacements[drone_idx]
    squared_distances = pairwise_squared_distances[drone_idx]

    neighbourhood_mask = (squared_distances > 0) & (squared_distances <= (DISTANCE_TRESHOLD + 2 * DRONE_SIZE)**2)
    neighbourhood_displacements = displacements[neighbourhood_mask]
    neighbourhood_indices = np.flatnonzero(neighbourhood_mask)
    neighbourhood_distances = squared_distances[neighbourhood_mask]
    contributions = np.sum(
        neighbourhood_displacements / neighbourhood_distances[:, None], 
        axis=0)

    return neighbourhood_indices, contributions


# def find_neighbours(drone_idx):
#     current_drone_pos = get_pos(drone_idx)
#     displacements = current_drone_pos - drone_pos

#     squared_distances = np.sum(displacements ** 2, axis=1)

#     neighbourhood_mask = (squared_distances > 0) & (squared_distances <= (DISTANCE_TRESHOLD + 2 * DRONE_SIZE)**2)
#     neighbourhood_displacements = displacements[neighbourhood_mask]
#     neighbourhood_indices = np.flatnonzero(neighbourhood_mask)
#     neighbourhood_distances = squared_distances[neighbourhood_mask]
#     contributions = np.sum(
#         neighbourhood_displacements / neighbourhood_distances[:, None], 
#         axis=0)

#     return neighbourhood_indices, contributions


def clamp_velocity(velocity):
    speed = np.linalg.norm(velocity)

    if speed == 0:
        return np.array([VELOCITY_MIN_LIMIT, 0.0])

    if speed > VELOCITY_MAX_LIMIT:
        return velocity / speed * VELOCITY_MAX_LIMIT

    if speed < VELOCITY_MIN_LIMIT:
        return velocity / speed * VELOCITY_MIN_LIMIT

    return velocity


def boids_algorithm(drone_idx, pairwise_displacements, pairwise_squared_distances):
    global neighbour_time_accumulator
    global averaging_timer_accumulator

    start_neighbours = time.perf_counter()
    drone_neighbours, separation_vector = find_neighbours(
        drone_idx, pairwise_displacements, pairwise_squared_distances
    )
    elapsed_neighbours = time.perf_counter() - start_neighbours
    neighbour_time_accumulator += elapsed_neighbours
    
    current_velocity = get_direction(drone_idx) * DRONE_VELOCITY
    current_position = get_pos(drone_idx)

    start_averaging = time.perf_counter()
    if len(drone_neighbours) > 0:
        neighbour_velocities = drone_directions[drone_neighbours] * DRONE_VELOCITY
        neighbour_positions = drone_pos[drone_neighbours]

        avg_neighbour_velocity = np.mean(neighbour_velocities, axis=0)
        avg_neighbour_pos = np.mean(neighbour_positions, axis=0)

        alignment_steering = avg_neighbour_velocity - current_velocity
        cohesion_contribution = avg_neighbour_pos - current_position
    else:
        alignment_steering = 0
        cohesion_contribution = 0
    elapsed_averaging = time.perf_counter() - start_averaging
    averaging_timer_accumulator += elapsed_averaging

    steering_contribution = (
        separation_vector * SEPARATION_STRENGTH
        + alignment_steering * ALIGNMENT_STRENGTH
        + cohesion_contribution * COHESION_STRENGTH)

    steering_contribution_length = np.sqrt(
        steering_contribution[0] ** 2 + steering_contribution[1] ** 2)
    
    if steering_contribution_length > STEERING_LIMIT:
        steering_contribution = (steering_contribution / steering_contribution_length) * STEERING_LIMIT

    return clamp_velocity(current_velocity + steering_contribution)


# def boids_algorithm(drone_idx):
#     global neighbour_time_accumulator
#     global averaging_timer_accumulator

#     start_neighbours = time.perf_counter()
#     drone_neighbours, separation_vector = find_neighbours(drone_idx)
#     elapsed_neighbours = time.perf_counter() - start_neighbours
#     neighbour_time_accumulator += elapsed_neighbours
    
#     current_velocity = get_direction(drone_idx) * DRONE_VELOCITY
#     current_position = get_pos(drone_idx)

#     start_averaging = time.perf_counter()
#     if len(drone_neighbours) > 0:
#         neighbour_velocities = drone_directions[drone_neighbours] * DRONE_VELOCITY
#         neighbour_positions = drone_pos[drone_neighbours]

#         avg_neighbour_velocity = np.mean(neighbour_velocities, axis=0)
#         avg_neighbour_pos = np.mean(neighbour_positions, axis=0)

#         alignment_steering = avg_neighbour_velocity - current_velocity
#         cohesion_contribution = avg_neighbour_pos - current_position
#     else:
#         alignment_steering = 0
#         cohesion_contribution = 0
#     elapsed_averaging = time.perf_counter() - start_averaging
#     averaging_timer_accumulator += elapsed_averaging

#     steering_contribution = (
#         separation_vector * SEPARATION_STRENGTH
#         + alignment_steering * ALIGNMENT_STRENGTH
#         + cohesion_contribution * COHESION_STRENGTH)

#     steering_contribution_length = np.sqrt(
#         steering_contribution[0] ** 2 + steering_contribution[1] ** 2)
    
#     if steering_contribution_length > STEERING_LIMIT:
#         steering_contribution = (steering_contribution / steering_contribution_length) * STEERING_LIMIT

#     return clamp_velocity(current_velocity + steering_contribution)


pygame.init()

screen = pygame.display.set_mode((X, Y))
pygame.display.set_caption("Drones")

font = pygame.font.Font("freesansbold.ttf", 32)
text = font.render("Collision", True, (255, 0, 0), (0, 0, 0))
textRect = text.get_rect()
textRect.center = (X // 2, Y // 2)

clock = pygame.time.Clock()
running = True

frame = 0

time_velocities = []
time_directions = []
time_positions = []
time_collisions = []
time_pairwise = []

exit_timer = True

start = time.perf_counter()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if (time.perf_counter() - start) >= 16 and exit_timer:
        running  = False
        

    screen.fill("white")

    new_velocities = []

    start_velocities = time.perf_counter()

    start_pairwise = time.perf_counter()
    pairwise_displacements = (
        drone_pos[:, None, :] - drone_pos[None, :, :]
    )

    pairwise_squared_distances = np.sum(
        pairwise_displacements ** 2, axis=2
    )

    elapsed_pairwise = time.perf_counter() - start_pairwise
    time_pairwise.append(elapsed_pairwise)

    neighbour_time_accumulator = 0
    averaging_timer_accumulator = 0

    for drone_idx in range(N_DRONES):
        new_velocities.append(
            boids_algorithm(
                drone_idx, 
                pairwise_displacements, 
                pairwise_squared_distances))
    
    neighbour_times.append(neighbour_time_accumulator)
    averaging_times.append(averaging_timer_accumulator)

    elapsed_velocities = time.perf_counter() - start_velocities
    time_velocities.append(elapsed_velocities)

    start_directions = time.perf_counter()
    for drone_idx in range(N_DRONES):
        new_direction = new_velocities[drone_idx] / DRONE_VELOCITY

        update_direction(
            drone_idx, 
            x_direction=new_direction[0], 
            y_direction=new_direction[1])

    elapsed_directions = time.perf_counter() - start_directions
    time_directions.append(elapsed_directions)

    start_positions = time.perf_counter()
    for drone_idx in range(N_DRONES):
        x_direction = drone_directions[drone_idx][0]
        y_direction = drone_directions[drone_idx][1]

        update_pos(drone_idx, x_direction, y_direction)
    elapsed_positions = time.perf_counter() - start_positions
    time_positions.append(elapsed_positions)

    start_collisions = time.perf_counter()
    collisions()
    elapsed_collisions = time.perf_counter() - start_collisions
    time_collisions.append(elapsed_collisions)

    for drone_idx in range(N_DRONES):
        pygame.draw.circle(screen, colors[drone_idx], drone_pos[drone_idx], DRONE_SIZE)

    pygame.display.flip()

    # clock.tick(60)
    frame += 1

pygame.quit()
elapsed = time.perf_counter() - start

print(f"Total: {elapsed * 1000} ms")
print(f"Velocities: {np.mean(time_velocities) * 1000} ms")
print(f"Directions: {np.mean(time_directions) * 1000} ms")
print(f"Positions: {np.mean(time_positions) * 1000} ms")
print(f"Collisions: {np.mean(time_collisions) * 1000} ms")
print(f"Neighbours: {np.mean(neighbour_times) * 1000} ms")
print(f"Averaging: {np.mean(averaging_times) * 1000} ms")
print(f"Pairwise: {np.mean(time_pairwise) * 1000} ms")
print()