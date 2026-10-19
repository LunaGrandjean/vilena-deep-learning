# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

## Apprentissage
- Époques enregistrées : 50.
- Modèle sélectionné : époque 34, perte de validation minimale 1.3934, accuracy de validation 46.7%.
- À cette époque : accuracy train 65.9%, écart train − validation +19.2%.
- Perte de validation finale : 1.6458.
- Signal compatible avec du surapprentissage : après le minimum de validation, la perte train baisse mais la perte de validation finit plus de 10 % au-dessus de ce minimum. À vérifier sur la courbe.
Les mesures train utilisent augmentation/dropout, celles de validation les désactivent : leur écart ne mesure pas à lui seul le surapprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 145.
- Accuracy test : 35.6%.
- F1 macro : 0.334.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.338 | 0.533 | 0.414 |
| Cubism | 0.179 | 0.111 | 0.137 |
| Impressionism | 0.421 | 0.178 | 0.250 |
| Pop_Art | 0.409 | 0.400 | 0.404 |
| Ukiyo_e | 0.397 | 0.556 | 0.463 |

Confusions les plus fréquentes (vrai → prédit) :
- Impressionism → Baroque : 23.
- Cubism → Baroque : 14.
- Cubism → Ukiyo_e : 13.
- Pop_Art → Ukiyo_e : 12.
- Cubism → Pop_Art : 9.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
