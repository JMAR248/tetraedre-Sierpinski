from fractions import Fraction
import numpy as np
from scipy.spatial import ConvexHull


def verifier_certificat_n11():
    v0 = np.array(
        ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=np.int64
    )
    n = 11
    nb_total = 1 << (2 * n)  # 4^11 = 4 194 304

    # 1. Generation des centres entiers dans l'ordre canonique en base 4
    centres = np.zeros((1, 3), dtype=np.int64)
    for step in range(n):
        decalage = 1 << (n - 1 - step)
        centres = (centres[:, None, :] + decalage * v0[None, :, :]).reshape(
            -1, 3
        )

    # 2. Chargement des intervalles [a, b] des 574 906 sous-tetraedres retires
    mask = np.ones(nb_total, dtype=bool)
    with open("certificat_intervalles_n11.txt", "r", encoding="utf-8") as f:
        for ligne in f:
            if ligne.startswith("#") or not ligne.strip():
                continue
            a_str, b_str = ligne.strip().split(",")
            mask[int(a_str) : int(b_str) + 1] = False

    centres_P = centres[mask]
    k = len(centres_P)

    # 3. Calcul exact du carre du diametre entier sur l'enveloppe convexe 3D
    r2 = np.sum(centres_P**2, axis=1)
    ext_centres = centres_P[r2 >= 3000000]
    sommets_ext = np.unique(
        (ext_centres[:, None, :] + v0[None, :, :]).reshape(-1, 3), axis=0
    )

    hull = ConvexHull(sommets_ext)
    h_pts = sommets_ext[hull.vertices]
    diffs = h_pts[:, None, :] - h_pts[None, :, :]
    d2_int = int(np.max(np.sum(diffs**2, axis=2)))

    # 4. Quotient rationnel exact Q_norm(P) = D_int^2 / (8 * k)
    q_exact = Fraction(d2_int, 8 * k)
    print(f"Sous-tetraedres retenus k : {k:,} / {nb_total:,}")
    print(f"Carre du diametre entier  : {d2_int}")
    print(f"Borne superieure exacte   : {q_exact} ({float(q_exact):.8f})")


if __name__ == "__main__":
    verifier_certificat_n11()
