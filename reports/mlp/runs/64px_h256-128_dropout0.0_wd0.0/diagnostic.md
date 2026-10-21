# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 2, perte de validation minimale 1.6491, accuracy de validation 38.2%.
- À cette époque : accuracy train 41.8%, écart train − validation +3.6%.
- Perte de validation finale : 3.6513.
- Signal compatible avec du surapprentissage : après le minimum de validation, la perte train baisse mais la perte de validation finit plus de 10 % au-dessus de ce minimum. À vérifier sur la courbe.
Les mesures train utilisent augmentation/dropout, celles de validation les désactivent : leur écart ne mesure pas à lui seul le surapprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 146.
- Accuracy test : 35.1%.
- F1 macro : 0.319.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.386 | 0.711 | 0.500 |
| Cubism | 0.154 | 0.044 | 0.069 |
| Impressionism | 0.227 | 0.222 | 0.225 |
| Pop_Art | 0.333 | 0.311 | 0.322 |
| Ukiyo_e | 0.488 | 0.467 | 0.477 |

Confusions les plus fréquentes (vrai → prédit) :
- Impressionism → Baroque : 22.
- Cubism → Baroque : 15.
- Pop_Art → Baroque : 11.
- Pop_Art → Impressionism : 11.
- Ukiyo_e → Pop_Art : 11.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
