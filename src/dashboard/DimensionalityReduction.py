import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import ClassicalMDS
from concurrent.futures import ThreadPoolExecutor
import os

def compute_distance_matrix(distance_fn, items):
    """
    Compute full pairwise distance matrix for items.

    Args:
        distance_fn: Function that computes distance between two items
        items: List of items

    Returns:
        D: numpy array of shape (n, n) with pairwise distances
    """
    n = len(items)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(distance_fn(items[i], items[j]))
    return D


def _compute_distance_row(args):
    """Helper function to compute one row of the distance matrix."""
    i, items, distance_fn = args
    n = len(items)
    row = np.zeros(n, dtype=float)
    for j in range(n):
        if i != j:
            row[j] = float(distance_fn(items[i], items[j]))
    return i, row


def compute_distance_matrix_parallel(distance_fn, items, parallel=None):
    """
    Compute full pairwise distance matrix for items with optional parallelization.

    Args:
        distance_fn: Function that computes distance between two items (must be picklable)
        items: List of items
        parallel: If None, no parallelization. If numeric, parallelize when n > parallel.

    Returns:
        D: numpy array of shape (n, n) with pairwise distances
    """
    n = len(items)
    D = np.zeros((n, n), dtype=float)

    use_parallel = parallel is not None and n > parallel

    if use_parallel:
        print(f'\033[33mUsing parallel distance matrix computation with {n} items\033[0m')
        n_workers = min(os.cpu_count() or 4, 50)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_compute_distance_row, (i, items, distance_fn)) for i in range(n)]
            for future in futures:
                i, row = future.result()
                D[i, :] = row
    else:
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = float(distance_fn(items[i], items[j]))

    return D


def _select_landmarks_random(n, n_landmarks, distance_fn, items, rng):
    """Random landmark selection."""
    return rng.choice(n, n_landmarks, replace=False)


def _select_landmarks_fps(n, n_landmarks, distance_fn, items, rng, r=1):
    """
    Furthest Point Sampling (FPS) landmark selection.

    Selects landmarks that are well-spread across the space by iteratively
    choosing points that are far from all existing landmarks.

    Args:
        n: Total number of items
        n_landmarks: Number of landmarks to select
        distance_fn: Function that computes distance between two items
        items: List of items
        rng: NumPy random generator
        r: Number of candidates to consider at each step (adds randomization)

    Returns:
        Array of landmark indices
    """
    # Choose first landmark uniformly at random
    first_landmark = rng.integers(0, n)
    S = [first_landmark]

    # Initialize minimum distances to first landmark
    m = np.array([float(distance_fn(items[i], items[first_landmark]))
                  if i != first_landmark else 0.0 for i in range(n)])

    for k in range(1, n_landmarks):
        # Find indices not in S
        available = np.array([i for i in range(n) if i not in S])
        m_available = m[available]

        # Get indices of r largest distances (furthest from all landmarks)
        r_actual = min(r, len(available))
        candidate_positions = np.argpartition(m_available, -r_actual)[-r_actual:]
        candidates = available[candidate_positions]

        # Choose uniformly at random from candidates
        new_landmark = rng.choice(candidates)
        S.append(new_landmark)

        # Update minimum distances
        for i in range(n):
            if i not in S:
                d_new = float(distance_fn(items[i], items[new_landmark]))
                m[i] = min(m[i], d_new)

    return np.array(S)


def landmark_mds(distance_fn, items, n_landmarks=None, landmark_method='random',
                 fps_candidates=1, random_state=42):
    """
    Landmark MDS: Efficient MDS approximation for large datasets.

    Args:
        distance_fn: Function that computes distance between two items
        items: List of items to embed
        n_landmarks: Number of landmarks to use (default: sqrt(n))
        landmark_method: Method for selecting landmarks ('random' or 'fps')
        fps_candidates: For FPS method, number of candidates to consider at each step
        random_state: Random seed for reproducibility

    Returns:
        XY: numpy array of shape (n, 2) with 2D coordinates
    """
    n = len(items)
    if n_landmarks is None:
        n_landmarks = min(max(20, int(np.sqrt(n))), n)

    # Ensure we don't have more landmarks than items
    n_landmarks = min(n_landmarks, n)

    # Select landmarks based on method
    rng = np.random.default_rng(random_state)
    if landmark_method == 'random':
        landmark_indices = _select_landmarks_random(n, n_landmarks, distance_fn, items, rng)
    elif landmark_method == 'fps':
        landmark_indices = _select_landmarks_fps(n, n_landmarks, distance_fn, items, rng,
                                                  r=fps_candidates)
    else:
        raise ValueError(f"Unknown landmark_method: {landmark_method}. Use 'random' or 'fps'.")

    # Compute n×k distance matrix (all points to landmarks)
    D_to_landmarks = np.zeros((n, n_landmarks))
    for i in range(n):
        for j, lm_idx in enumerate(landmark_indices):
            if i != lm_idx:
                D_to_landmarks[i, j] = float(distance_fn(items[i], items[lm_idx]))

    # Extract k×k landmark-to-landmark distance matrix
    D_landmarks = D_to_landmarks[landmark_indices, :]

    # Run MDS on landmarks only
    # mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=random_state)
    # XY_landmarks = mds.fit_transform(D_landmarks)
    cmds = ClassicalMDS(n_components=2, metric="precomputed")
    XY_landmarks = cmds.fit_transform(D_landmarks)

    # Initialize output coordinates
    XY = np.zeros((n, 2))
    XY[landmark_indices] = XY_landmarks

    # Out-of-sample extension via lateration (least-squares distance matching)
    landmark_set = set(landmark_indices)
    X_L = XY_landmarks  # Landmark positions in 2D

    # Precompute squared norms of landmark positions
    X_L_sq_norms = np.sum(X_L ** 2, axis=1)

    for i in range(n):
        if i in landmark_set:
            continue

        # Distances from point i to all landmarks
        delta = D_to_landmarks[i, :]
        delta_sq = delta ** 2

        # Use landmark 0 as reference
        # Build system: A @ x = b
        # A[a-1, :] = 2 * (X_L[a] - X_L[0])
        # b[a-1] = ||X_L[a]||² - ||X_L[0]||² - (δ[a]² - δ[0]²)
        A = 2 * (X_L[1:] - X_L[0])
        b = X_L_sq_norms[1:] - X_L_sq_norms[0] - (delta_sq[1:] - delta_sq[0])

        # Solve least-squares problem: argmin_x ||Ax - b||²
        XY[i], residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    return XY