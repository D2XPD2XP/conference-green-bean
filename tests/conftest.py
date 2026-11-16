"""Konfigurasi pytest bersama."""

import matplotlib

# Test hanya menyimpan figur ke tmp_path; backend non-interaktif mencegah
# jendela GUI terbuka. Diset di sini, bukan di modul library, agar backend
# inline Jupyter tidak ikut tertimpa.
matplotlib.use("Agg")
