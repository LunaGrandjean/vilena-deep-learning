# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 27, perte de validation minimale 1.3631, accuracy de validation 44.9%.
- À cette époque : accuracy train 49.0%, écart train − validation +4.1%.
- Perte de validation finale : 1.4145.
- Le critère automatique de divergence (>10 % de hausse de perte validation et baisse de perte train après le minimum) n'est pas rempli. Cela ne prouve pas l'absence de surapprentissage.
Les mesures train utilisent augmentation/dropout, celles de validation les désactivent : leur écart ne mesure pas à lui seul le surapprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 139.
- Accuracy test : 38.2%.
- F1 macro : 0.363.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.489 | 0.511 | 0.500 |
| Cubism | 0.138 | 0.089 | 0.108 |
| Impressionism | 0.387 | 0.267 | 0.316 |
| Pop_Art | 0.436 | 0.378 | 0.405 |
| Ukiyo_e | 0.380 | 0.667 | 0.484 |

Confusions les plus fréquentes (vrai → prédit) :
- Cubism → Ukiyo_e : 18.
- Pop_Art → Ukiyo_e : 13.
- Impressionism → Baroque : 12.
- Impressionism → Ukiyo_e : 11.
- Cubism → Baroque : 8.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
