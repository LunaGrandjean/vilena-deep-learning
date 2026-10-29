# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 49, perte de validation minimale 0.5067, accuracy de validation 79.6%.
- À cette époque : accuracy train 96.9%, écart train − validation +17.3%.
- Perte de validation finale : 0.5070.
- Le critère automatique de divergence (>10 % de hausse de perte validation et baisse de perte train après le minimum) n'est pas rempli. Cela ne prouve pas l'absence de surapprentissage.
Les mesures train utilisent le retournement horizontal aléatoire, sans dropout ; la validation n'est pas augmentée. L'écart ne mesure pas à lui seul le surapprentissage.
- VGG16 : couches ImageNet gelées, seule la sortie linéaire est apprise ; dropout préentraîné désactivé, retournement horizontal aléatoire sur train uniquement.
- Paramètres entraînés : 20,485 ; extraction initiale des représentations : 656.72 s, comprise dans le temps d'entraînement. Le benchmark CPU exécute le réseau complet sans cache.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 44.
- Accuracy test : 80.4%.
- F1 macro : 0.805.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.850 | 0.756 | 0.800 |
| Cubism | 0.822 | 0.822 | 0.822 |
| Impressionism | 0.731 | 0.844 | 0.784 |
| Pop_Art | 0.756 | 0.756 | 0.756 |
| Ukiyo_e | 0.884 | 0.844 | 0.864 |

Confusions les plus fréquentes (vrai → prédit) :
- Baroque → Impressionism : 7.
- Cubism → Pop_Art : 5.
- Pop_Art → Impressionism : 5.
- Impressionism → Baroque : 5.
- Pop_Art → Cubism : 4.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
