# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 24, perte de validation minimale 0.9776, accuracy de validation 63.6%.
- À cette époque : accuracy train 63.3%, écart train − validation -0.2%.
- Perte de validation finale : 1.1633.
- Signal compatible avec du surapprentissage : après le minimum de validation, la perte train baisse mais la perte de validation finit plus de 10 % au-dessus de ce minimum. À vérifier sur la courbe.
Les mesures train utilisent augmentation/dropout, celles de validation les désactivent : leur écart ne mesure pas à lui seul le surapprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 88.
- Accuracy test : 60.9%.
- F1 macro : 0.602.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.593 | 0.778 | 0.673 |
| Cubism | 0.429 | 0.400 | 0.414 |
| Impressionism | 0.538 | 0.467 | 0.500 |
| Pop_Art | 0.649 | 0.533 | 0.585 |
| Ukiyo_e | 0.812 | 0.867 | 0.839 |

Confusions les plus fréquentes (vrai → prédit) :
- Impressionism → Baroque : 16.
- Pop_Art → Cubism : 15.
- Cubism → Pop_Art : 8.
- Cubism → Impressionism : 7.
- Baroque → Impressionism : 6.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
