import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def generar_scatter_rank_shap_vs_rf():
    shap_data = {
        'Init_Win_bytes_forward': 8.3928, 'Init_Win_bytes_backward': 6.5360, 'Bwd Packet Length Min': 4.2550,
        'Fwd Header Length': 3.7298, 'Bwd Packet Length Std': 3.2686, 'Bwd Packet Length Mean': 3.1010,
        'Avg Bwd Segment Size': 2.9712, 'Bwd Packet Length Max': 2.7605, 'Packet Length Mean': 2.6838,
        'Max Packet Length': 2.6375, 'Total Length of Bwd Packets': 2.5535, 'Flow IAT Mean': 2.5033,
        'Average Packet Size': 2.4732, 'Packet Length Std': 2.3066, 'Bwd Header Length': 2.2939,
        'Bwd Packets/s': 2.2839, 'Packet Length Variance': 2.2372, 'Fwd Packet Length Max': 2.1428,
        'Total Length of Fwd Packets': 2.0568, 'Fwd IAT Std': 1.9867, 'Flow IAT Std': 1.9673,
        'Flow IAT Max': 1.9228, 'Flow Duration': 1.9117, 'Flow Packets/s': 1.8124,
        'Fwd Packets/s': 1.7854, 'Fwd Packet Length Mean': 1.7652, 'Avg Fwd Segment Size': 1.7645,
        'Flow IAT Min': 1.7588, 'PSH Flag Count': 1.6735, 'Fwd IAT Max': 1.6446,
        'Fwd IAT Mean': 1.5650, 'Flow Bytes/s': 1.5266, 'Fwd IAT Total': 1.5128,
        'Fwd Packet Length Std': 1.4026, 'Fwd IAT Min': 1.3360, 'Total Backward Packets': 1.2937,
        'Min Packet Length': 1.2232, 'Total Fwd Packets': 1.1009, 'Bwd IAT Total': 0.9943,
        'Bwd IAT Mean': 0.9765, 'act_data_pkt_fwd': 0.9483, 'Bwd IAT Std': 0.9120,
        'Bwd IAT Max': 0.9079, 'ACK Flag Count': 0.8157, 'Bwd IAT Min': 0.7724,
        'Fwd Packet Length Min': 0.7691, 'Down/Up Ratio': 0.3240, 'SYN Flag Count': 0.3008,
        'FIN Flag Count': 0.1381, 'RST Flag Count': 0.0000
    }

    rf_data = {
        'Init_Win_bytes_backward': 8.3396, 'Init_Win_bytes_forward': 3.6193, 'Bwd Packets/s': 3.4610,
        'Flow IAT Mean': 3.2557, 'Fwd Packet Length Max': 2.9707, 'Total Length of Bwd Packets': 2.9061,
        'Fwd Packet Length Std': 2.8566, 'Average Packet Size': 2.8139, 'Packet Length Mean': 2.7374,
        'Flow Packets/s': 2.7189, 'Max Packet Length': 2.7059, 'Total Length of Fwd Packets': 2.6851,
        'Bwd Packet Length Max': 2.6281, 'Bwd Packet Length Mean': 2.5263, 'Flow IAT Max': 2.3532,
        'Flow Duration': 2.3452, 'Bwd Header Length': 2.3285, 'Flow IAT Std': 2.2798,
        'Bwd Packet Length Min': 2.2681, 'Avg Bwd Segment Size': 2.2335, 'Fwd IAT Max': 2.2220,
        'Avg Fwd Segment Size': 2.1699, 'Fwd IAT Std': 2.1493, 'Fwd Packets/s': 2.0946,
        'Packet Length Variance': 2.0895, 'Fwd IAT Mean': 2.0204, 'Fwd IAT Total': 1.9693,
        'Flow Bytes/s': 1.9657, 'Fwd Header Length': 1.9392, 'Bwd Packet Length Std': 1.9365,
        'Fwd Packet Length Mean': 1.8745, 'Packet Length Std': 1.8602, 'Total Backward Packets': 1.5696,
        'Fwd IAT Min': 1.5512, 'Total Fwd Packets': 1.4964, 'Flow IAT Min': 1.4816,
        'Bwd IAT Min': 0.9952, 'act_data_pkt_fwd': 0.9864, 'Bwd IAT Mean': 0.9488,
        'PSH Flag Count': 0.9415, 'Bwd IAT Max': 0.9026, 'Min Packet Length': 0.8594,
        'Bwd IAT Total': 0.8241, 'Fwd Packet Length Min': 0.6438, 'SYN Flag Count': 0.6405,
        'ACK Flag Count': 0.6296, 'Down/Up Ratio': 0.3411, 'FIN Flag Count': 0.0593,
        'RST Flag Count': 0.0000
    }

    # Calcular rankings (1 = mayor porcentaje)
    features = list(set(shap_data.keys()) & set(rf_data.keys()))

    shap_sorted = sorted(features, key=lambda f: shap_data[f], reverse=True)
    rf_sorted   = sorted(features, key=lambda f: rf_data[f],   reverse=True)

    shap_rank = {f: i + 1 for i, f in enumerate(shap_sorted)}
    rf_rank   = {f: i + 1 for i, f in enumerate(rf_sorted)}

    x_vals = [shap_rank[f] for f in features]
    y_vals = [rf_rank[f]   for f in features]

    r2 = r2_score(x_vals, y_vals)
    print(f"R² (rankings): {r2:.4f}")

    # Scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, color='royalblue', alpha=0.8, edgecolors='black', s=60)

    # Etiquetar puntos con mayor discrepancia o más importantes
    for f in features:
        x, y = shap_rank[f], rf_rank[f]
        if x <= 5 or y <= 5 or abs(x - y) >= 20:
            plt.annotate(
                f,
                (x, y),
                textcoords="offset points",
                xytext=(8, 5),
                ha='left',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
            )

    # Línea de concordancia perfecta (rank SHAP = rank RF)
    max_rank = len(features)
    plt.plot([1, max_rank], [1, max_rank], color='crimson', linestyle='--',
             label='Línea de Concordancia (Rank SHAP = Rank RF)')

    plt.title(f'Comparativa de Rankings: SHAP vs Random Forest (R² = {r2:.2f})',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Ranking según SHAP (1 = más importante)', fontsize=12)
    plt.ylabel('Ranking según Random Forest MDI (1 = más importante)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, max_rank + 1)
    plt.ylim(0, max_rank + 1)
    plt.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig('shap_vs_rf_rank.pdf', dpi=300)
    print("[*] Gráfico guardado como 'shap_vs_rf_rank.pdf'")

if __name__ == "__main__":
    generar_scatter_rank_shap_vs_rf()