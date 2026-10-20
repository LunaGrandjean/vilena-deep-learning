# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 49, perte de validation minimale 0.4054, accuracy de validation 86.7%.
- À cette époque : accuracy train 100.0%, écart train − validation +13.3%.
- Perte de validation finale : 0.4152.
- Le critère automatique de divergence (>10 % de hausse de perte validation et baisse de perte train après le minimum) n'est pas rempli. Cela ne prouve pas l'absence de surapprentissage.
Les mesures train utilisent augmentation/dropout, celles de validation les désactivent : leur écart ne mesure pas à lui seul le surapprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 43.
- Accuracy test : 80.9%.
- F1 macro : 0.809.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.825 | 0.733 | 0.776 |
| Cubism | 0.830 | 0.867 | 0.848 |
| Impressionism | 0.745 | 0.844 | 0.792 |
| Pop_Art | 0.795 | 0.778 | 0.787 |
| Ukiyo_e | 0.860 | 0.822 | 0.841 |

Confusions les plus fréquentes (vrai → prédit) :
- Baroque → Impressionism : 7.
- Impressionism → Baroque : 5.
- Pop_Art → Cubism : 4.
- Cubism → Pop_Art : 3.
- Baroque → Pop_Art : 3.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
