# Diagnostic de l'expérience

Généré à partir des mesures de cette exécution.

Pas d'historique fourni : aucun constat sur les courbes d'apprentissage.

## Résultats et erreurs
- Images de test : 225 ; erreurs : 131.
- Accuracy test : 41.8%.
- F1 macro : 0.398.

| Style | Précision | Rappel | F1 |
|---|---:|---:|---:|
| Baroque | 0.492 | 0.667 | 0.566 |
| Cubism | 0.206 | 0.156 | 0.177 |
| Impressionism | 0.353 | 0.267 | 0.304 |
| Pop_Art | 0.441 | 0.333 | 0.380 |
| Ukiyo_e | 0.484 | 0.667 | 0.561 |

Confusions les plus fréquentes (vrai → prédit) :
- Impressionism → Baroque : 15.
- Cubism → Ukiyo_e : 12.
- Pop_Art → Cubism : 10.
- Cubism → Impressionism : 9.
- Impressionism → Ukiyo_e : 9.

La galerie montre les erreurs ayant les probabilités prédites les plus élevées. Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel doit compléter les comptages avant d'attribuer une cause aux erreurs.

## Limites
La séparation est stratifiée par image, pas par artiste. Des tableaux d'un même peintre peuvent apparaître dans train et test. Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. Les erreurs test servent au bilan final ; les réglages doivent rester fondés sur la validation, sans réutiliser le test pour choisir un modèle.
