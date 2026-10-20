# Graphiques du diaporama

Six figures PNG haute résolution, avec fond transparent et une palette bordeaux
et olive. Elles reprennent les résultats enregistrés, sans nouvel entraînement.
Conserver leurs proportions dans Canva pour garder les axes et légendes visibles.

| Fichier | Slide | Données sources |
|---|---|---|
| [mlp_regularisation.png](mlp_regularisation.png) | MLP regularisation study | Historiques `history_64px_h256-128_dropout0.0_wd0.0.json` et `history_64px_h256-128_dropout0.5_wd0.0.json` dans `reports/mlp/` |
| [cnn_learning_curves.png](cnn_learning_curves.png) | Small CNN: learning from scratch | `outputs/comparaison_v1/cnn/history.json` (historique local, ignoré par Git) |
| [resnet18_learning_curves.png](resnet18_learning_curves.png) | ResNet18: transfer learning | `outputs/comparaison_v1/resnet18/history.json` (historique local, ignoré par Git) |
| [accuracy_vs_cpu.png](accuracy_vs_cpu.png) | Classification quality and CPU cost | `reports/comparaison_v1/comparison.csv`, uniquement régression logistique, CNN et ResNet18 |
| [mlp_confusion_matrix.png](mlp_confusion_matrix.png) | Error analysis — Final MLP | `reports/mlp/test_metrics_64px_h256-128_dropout0.5_wd0.0.json` |
| [resnet18_confusion_matrix.png](resnet18_confusion_matrix.png) | Error analysis — ResNet18 | `reports/comparaison_v1/resnet18/confusion_matrix.csv` |

Les courbes CNN et ResNet18 utilisent le bordeaux pour l'entraînement et l'olive
pour la validation. Les pointillés marquent les checkpoints sélectionnés par
minimum de perte de validation : époque 24 pour le CNN, époque 49 pour ResNet18.
Les deux matrices affichent des effectifs, avec 45 images par style et la même
échelle de couleurs de 0 à 45. Lignes : style réel ; colonnes : style prédit.

Le MLP historique à dropout 0,5 atteint 36,0 % (81/225), sur un partage
1 049/226/225 différent du partage commun 1 050/225/225. Il n'a pas de mesure CPU.
Le graphique CPU exclut donc ce MLP et n'utilise pas la latence de l'ancien MLP
32 px. Les temps CPU proviennent de l'ordinateur de développement.

Les chemins absolus présents dans les métadonnées historiques des rapports
décrivent leur emplacement au moment de l'exécution. Les rapports sont conservés
tels quels après le déplacement du projet à la racine du dépôt.
