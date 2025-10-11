cp src/evaluate.py src/evaluate.backup2.py 2>/dev/null || true

python - <<'PY'
from pathlib import Path
p = Path("src/evaluate.py")
code = p.read_text()
# вставка нормализованной CM, если её нет
if "confusion_matrix_norm.png" not in code:
    code = code.replace("fig.savefig(os.path.join(args.out_dir, \"confusion_matrix.png\"), bbox_inches=\"tight\")\n    plt.close(fig)\n",
                        "fig.savefig(os.path.join(args.out_dir, \"confusion_matrix.png\"), bbox_inches=\"tight\")\n    plt.close(fig)\n\n"
                        "# --- normalized CM ---\n"
                        "fig = plt.figure()\n"
                        "disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, normalize='true'), display_labels=le.classes_)\n"
                        "disp.plot(values_format='.2f')\n"
                        "fig.savefig(os.path.join(args.out_dir, 'confusion_matrix_norm.png'), bbox_inches='tight')\n"
                        "plt.close(fig)\n")
# вставка PR/ROC блока с поддержкой SVM decision_function, если его нет
if "pr_curve.png" not in code:
    code += ("\n# --- ROC/PR block (proba or decision_function) ---\n"
             "scores = None\n"
             "if hasattr(model.named_steps['clf'], 'predict_proba') and len(le.classes_) == 2:\n"
             "    scores = model.predict_proba(X_test)[:, 1]\n"
             "elif hasattr(model.named_steps['clf'], 'decision_function') and len(le.classes_) == 2:\n"
             "    scores = model.decision_function(X_test)\n"
             "if scores is not None:\n"
             "    fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)\n"
             "    auc = roc_auc_score(y_test, scores)\n"
             "    fig = plt.figure()\n"
             "    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')\n"
             "    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC (AUC={auc:.3f})')\n"
             "    fig.savefig(os.path.join(args.out_dir, 'roc_curve.png'), bbox_inches='tight'); plt.close(fig)\n"
             "    from sklearn.metrics import precision_recall_curve, average_precision_score\n"
             "    precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=1)\n"
             "    ap = average_precision_score(y_test, scores)\n"
             "    fig = plt.figure(); plt.step(recall, precision, where='post')\n"
             "    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR curve (AP={ap:.3f})')\n"
             "    fig.savefig(os.path.join(args.out_dir, 'pr_curve.png'), bbox_inches='tight'); plt.close(fig)\n")
p.write_text(code)
print("evaluate.py patched")
PY

make evaluate

