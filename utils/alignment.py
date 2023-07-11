def get_two_alignment(
    seq1, seq2, gap_penalty=-1, match_score=1, mismatch_penalty=-5, dummychar="-"
):
    # Initialize the scoring matrix
    def create_matrix(dimensions):
        rows, cols = dimensions
        return [[0] * cols for _ in range(rows)]

    rows = len(seq1) + 1
    cols = len(seq2) + 1
    scores = create_matrix((rows, cols))
    pointers = create_matrix((rows, cols))
    alignment = []

    # Fill the first row and column with gap penalties
    for i in range(rows):
        scores[i][0] = i * gap_penalty
        pointers[i][0] = (i - 1, 0)
    for j in range(cols):
        scores[0][j] = j * gap_penalty
        pointers[0][j] = (0, j - 1)

    # Calculate the scores and pointers for each cell
    for i in range(1, rows):
        for j in range(1, cols):
            match = scores[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
            )
            delete = scores[i - 1][j] + gap_penalty
            insert = scores[i][j - 1] + gap_penalty
            scores[i][j] = max(match, delete, insert)
            if scores[i][j] == match:
                pointers[i][j] = (i - 1, j - 1)
            elif scores[i][j] == delete:
                pointers[i][j] = (i - 1, j)
            else:
                pointers[i][j] = (i, j - 1)

    # Traceback to construct the alignment
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        di, dj = pointers[i][j]
        if di == i - 1 and dj == j - 1:
            aligned_chars = (seq1[i - 1], seq2[j - 1])
        elif di == i - 1 and dj == j:
            aligned_chars = (seq1[i - 1], dummychar)
        else:
            aligned_chars = (dummychar, seq2[j - 1])
        alignment.append(aligned_chars)
        i, j = di, dj

    return alignment[::-1], scores


def get_three_alignment(
    seq1,
    seq2,
    seq3,
    gap_penalty=-1,
    match_score=1,
    half_match_score=0.5,
    mismatch_score=-100,
    half_mismatch_penalty=-50,
    dummychar="-",
):
    def create_matrix(dimensions):
        rows, cols, depth = dimensions
        return [[[0] * depth for _ in range(cols)] for _ in range(rows)]

    # Initialize the scoring matrix
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    depth = len(seq3) + 1
    scores = create_matrix((rows, cols, depth))
    pointers = create_matrix((rows, cols, depth))
    alignments = []

    def get_aligned_word(i):
        return "".join([chars[i] for chars in alignment])

    # Fill the first row, column, and depth with gap penalties
    for i in range(rows):
        scores[i][0][0] = i * gap_penalty
        pointers[i][0][0] = (i - 1, 0, 0)
    scores_jk = get_two_alignment(
        seq2,
        seq3,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for j in range(cols):
        for k in range(depth):
            scores[0][j][k] = scores_jk[j][k]

    for j in range(cols):
        scores[0][j][0] = j * gap_penalty
        pointers[0][j][0] = (0, j - 1, 0)
    scores_ik = get_two_alignment(
        seq1,
        seq3,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for i in range(rows):
        for k in range(depth):
            scores[i][0][k] = scores_ik[i][k]

    for k in range(depth):
        scores[0][0][k] = k * gap_penalty
        pointers[0][0][k] = (0, 0, k - 1)
    scores_ij = get_two_alignment(
        seq1,
        seq2,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for i in range(rows):
        for j in range(cols):
            scores[i][j][0] = scores_ij[i][j]

    # Calculate the scores and pointers for each cell
    for i in range(1, rows):
        for j in range(1, cols):
            for k in range(1, depth):
                match = scores[i - 1][j - 1][k - 1] + (
                    match_score
                    if seq1[i - 1] == seq2[j - 1] == seq3[k - 1]
                    else mismatch_score
                )
                match_ij = scores[i - 1][j - 1][k] + (
                    half_match_score
                    if seq1[i - 1] == seq2[j - 1]
                    else half_mismatch_penalty
                )
                match_ik = scores[i - 1][j][k - 1] + (
                    half_match_score
                    if seq1[i - 1] == seq3[k - 1]
                    else half_mismatch_penalty
                )
                match_jk = scores[i][j - 1][k - 1] + (
                    half_match_score
                    if seq2[j - 1] == seq3[k - 1]
                    else half_mismatch_penalty
                )
                insert_i = scores[i - 1][j][k] + gap_penalty
                insert_j = scores[i][j - 1][k] + gap_penalty
                insert_k = scores[i][j][k - 1] + gap_penalty
                scores[i][j][k] = max(
                    match, insert_i, insert_j, insert_k, match_ij, match_ik, match_jk
                )
                if scores[i][j][k] == match:
                    pointers[i][j][k] = (i - 1, j - 1, k - 1)
                elif scores[i][j][k] == match_ij:
                    pointers[i][j][k] = (i - 1, j - 1, k)
                elif scores[i][j][k] == match_ik:
                    pointers[i][j][k] = (i - 1, j, k - 1)
                elif scores[i][j][k] == match_jk:
                    pointers[i][j][k] = (i, j - 1, k - 1)
                elif scores[i][j][k] == insert_i:
                    pointers[i][j][k] = (i - 1, j, k)
                elif scores[i][j][k] == insert_j:
                    pointers[i][j][k] = (i, j - 1, k)
                else:
                    pointers[i][j][k] = (i, j, k - 1)

    # Traceback to construct the alignments
    i, j, k = rows - 1, cols - 1, depth - 1
    while i > 0 or j > 0 or k > 0:
        if sum([i == 0, j == 0, k == 0]) == 1:
            if i == 0:
                align_2 = get_two_alignment(
                    seq2[:j], seq3[:k], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (dummychar, seq2_ch, seq3_ch) for seq2_ch, seq3_ch in align_2
                ]
            elif j == 0:
                align_2 = get_two_alignment(
                    seq1[:i], seq3[:k], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (seq1_ch, dummychar, seq3_ch) for seq1_ch, seq3_ch in align_2
                ]
            else:
                align_2 = get_two_alignment(
                    seq1[:i], seq2[:j], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (seq1_ch, seq2_ch, dummychar) for seq1_ch, seq2_ch in align_2
                ]
            align_2.extend(alignments[::-1])
            alignment = align_2
            return get_aligned_word(0), get_aligned_word(1), get_aligned_word(2)
        di, dj, dk = pointers[i][j][k]
        if di == i - 1 and dj == j - 1 and dk == k - 1:
            alignments.append((seq1[i - 1], seq2[j - 1], seq3[k - 1]))
        elif di == i - 1 and dj == j - 1 and dk == k:
            alignments.append((seq1[i - 1], seq2[j - 1], dummychar))
        elif di == i - 1 and dj == j and dk == k - 1:
            alignments.append((seq1[i - 1], dummychar, seq3[k - 1]))
        elif di == i and dj == j - 1 and dk == k - 1:
            alignments.append((dummychar, seq2[j - 1], seq3[k - 1]))
        elif di == i - 1 and dj == j and dk == k:
            alignments.append((seq1[i - 1], dummychar, dummychar))
        elif di == i and dj == j - 1 and dk == k:
            alignments.append((dummychar, seq2[j - 1], dummychar))
        else:
            alignments.append((dummychar, dummychar, seq3[k - 1]))
        i, j, k = di, dj, dk

    alignment = alignments[::-1]
    return get_aligned_word(0), get_aligned_word(1), get_aligned_word(2)


# ----------------recover-a-string-from-three-candidates--------
def recover_from_aligned_candidates(
    textract_line_confidence,
    aligned_s1,
    aligned_s2,
    aligned_s3,
    dummychar="-",
):
    def get_char_if_not_dummy(ch):
        return ch if ch != dummychar else ""

    recovered_string = ""
    for c1, c2, c3 in zip(aligned_s1, aligned_s2, aligned_s3):
        if c1 == c2:
            recovered_string += get_char_if_not_dummy(c1)
        elif c1 == c3:
            recovered_string += get_char_if_not_dummy(c1)
        elif c2 == c3:
            recovered_string += get_char_if_not_dummy(c2)
        else:
            # No agreement, first go with Textract (c1) as we know our accuracy level with Textract
            if c1 != dummychar:
                recovered_string += c1
            elif c2 != dummychar:
                recovered_string += c2
            else:
                recovered_string += c3

    new_line_confidence = 0
    match_proportion = 1
    for c1, c2, c3 in zip(aligned_s1, aligned_s2, aligned_s3):
        # If all different, then use textract line confidence
        if c1 != c2 and c2 != c3 and c1 != c3:
            # cut confidence boost in half
            match_proportion *= 1 / 2

        # if all agree, then for that character maintain match proportion
        elif (c1 == c2) and (c2 == c3) and (c1 == c3):
            # maintain full confidence boost
            match_proportion *= 1

        else:  # two of three agree, reduce match proportion to 2/3 previously level
            # cut confidence boost by 10%
            match_proportion *= (9) / 10
    confidence_boost = (100 - textract_line_confidence) * (match_proportion)
    new_line_confidence = textract_line_confidence + confidence_boost

    return recovered_string, new_line_confidence



from itertools import product
from collections import deque


def needleman_wunsch(x, y, gap: str = "-"):
    """Run the Needleman-Wunsch algorithm on two sequences.

    x, y -- sequences.

    Code based on pseudocode in Section 3 of:

    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf
    """
    N, M = len(x), len(y)
    s = lambda a, b: int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1
    aligned_x = "".join(gap if i is None else x[i] for i, _ in alignment)
    aligned_y = "".join(gap if j is None else y[j] for _, j in alignment)
    return list(aligned_x), list(aligned_y)


def align(s1, s2, s3):
    # align s1 to s2 and to s3
    aligned_s1_12, _ = needleman_wunsch(s1, s2, gap="+")
    aligned_s1_13, _ = needleman_wunsch(s1, s3, gap="+")
    # align s2 to s1 and s3
    aligned_s2_21, _ = needleman_wunsch(s2, s1, gap="+")
    aligned_s2_23, _ = needleman_wunsch(s2, s3, gap="+")
    # align s3 to s1 and s2
    aligned_s3_31, _ = needleman_wunsch(s3, s1, gap="+")
    aligned_s3_32, _ = needleman_wunsch(s3, s2, gap="+")

    # Next, 'self-align', that is, align the two distinct alignments we obtained above for each string
    # self-align s1
    self_aligned_s1, _ = needleman_wunsch(
        "".join(aligned_s1_12), "".join(aligned_s1_13)
    )
    # self-align s2
    self_aligned_s2, _ = needleman_wunsch(
        "".join(aligned_s2_21), "".join(aligned_s2_23)
    )
    # self-align s3
    self_aligned_s3, _ = needleman_wunsch(
        "".join(aligned_s3_31), "".join(aligned_s3_32)
    )

    # there will be a mix of '+' and '-' characters, we replace all '+' with '-' so we can continue using previous code downstream
    self_aligned_s1 = "".join(self_aligned_s1).replace("+", "-")
    self_aligned_s2 = "".join(self_aligned_s2).replace("+", "-")
    self_aligned_s3 = "".join(self_aligned_s3).replace("+", "-")
    return self_aligned_s1, self_aligned_s2, self_aligned_s3

