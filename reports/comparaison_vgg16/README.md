# VGG16 : transfert avec extracteur gelé

## Résultats mesurés

| Mesure | VGG16 gelé | ResNet18 existant |
|---|---:|---:|
| Bonnes prédictions test | 181/225 | 182/225 |
| Accuracy test | 80,4 % | 80,9 % |
| F1 macro | 0,805 | 0,809 |
| CPU modèle seul, médiane | 296,706 ms | 43,204 ms |
| CPU chaîne complète, médiane | 305,282 ms | 42,510 ms |
| Taille du checkpoint | 537,14 Mo | 44,79 Mo |

VGG16 atteint un score proche de ResNet18, avec une seule image correctement
classée en moins : cet écart ne démontre pas une différence statistiquement
significative de précision. En revanche, dans ces mesures, sa latence complète
est environ **7,2 fois supérieure** et son checkpoint **12 fois plus volumineux**.
Pour la borne, ce premier essai ne donne donc pas d'avantage pratique à VGG16
par rapport au pipeline ResNet18 existant, sous les conditions mesurées.

Le checkpoint est choisi à l'époque **49** : perte validation **0,5067**, accuracy
validation **79,6 %**, accuracy train **96,9 %**. Cet écart de généralisation
reste à discuter ; il ne suffit pas à lui seul à expliquer les erreurs.
Les 50 époques ont été exécutées, sans arrêt anticipé.

L'entraînement CPU total dure **658,30 s** (environ 11 minutes), dont
**656,72 s** d'extraction initiale des représentations. Les époques de la petite
sortie sont rapides parce que les couches ImageNet ne sont pas réentraînées.
Cette durée ne mesure donc pas le coût d'un fine-tuning complet de VGG16.

Vérifications : les couches gelées du checkpoint sont identiques aux poids
ImageNet officiels ; le rechargement dans la démo `wikiart.predict` reproduit
les probabilités du CSV test pour l'image de contrôle `Pop_Art/s71_r0376.jpg`.
Les sept tests automatisés du projet passent, dont le test du cache de
représentations, du gel des paramètres et du checkpoint VGG16.

## Protocole

- Données : les mêmes 1 500 images WikiArt (300 par style), séparées en
  1 050 images train, 225 validation et 225 test ; graine 0.
- Entrée : RGB 224 × 224, redimensionnement à 256 puis crop central et
  normalisation ImageNet, comme le pipeline ResNet18 du projet.
- Modèle : VGG16 Torchvision, poids `VGG16_Weights.IMAGENET1K_V1`.
  Les 13 convolutions et les deux couches de 4 096 neurones sont conservées
  et gelées. Seule la nouvelle couche finale `Linear(4096, 5)` est apprise,
  soit **20 485 paramètres entraînables**.
- Apprentissage : Adam, taux 0,0001, 50 époques, batch 32, CPU avec un thread.
  Sélection par **perte de validation minimale** ; aucun réglage sur le test.
- Augmentation : retournement horizontal aléatoire de probabilité 0,5 sur
  train uniquement. Les représentations des deux orientations train et des
  images validation originales sont calculées une fois en mémoire. À chaque
  accès train, une orientation est choisie aléatoirement. Le dropout des
  couches gelées reste désactivé pour rendre cette extraction déterministe.
  Le test n'entre pas dans ce cache et est évalué après sélection du checkpoint.
- Durée : le temps d'entraînement comprend l'extraction des représentations,
  les 50 époques, leurs validations et l'écriture du checkpoint complet.
  La préparation initiale (construction/chargement du modèle et données) est
  comptabilisée séparément. Le temps d'extraction est détaillé dans les métadonnées.
- Benchmark : **réseau VGG16 complet**, une image à la fois, un thread CPU,
  cinq passages de chauffe, 50 répétitions. Modèle seul et chaîne complète
  (JPEG, prétraitement, prédiction) sont mesurés séparément. Aucun cache de
  représentations n'est utilisé pour ces mesures.

## Fichiers

- [Métriques test](vgg16/test_metrics.json), [benchmark CPU](vgg16/cpu_benchmark.json).
- [Courbes d'apprentissage](vgg16/learning_curves.png),
  [diagnostic](vgg16/diagnostic.md), [matrice de confusion](vgg16/confusion_matrix.png).
- [Comparaison avec MLP, CNN, ResNet18 et régression logistique](comparison.md)
  et [tableau CSV](comparison.csv).
- Checkpoint local : `checkpoints/comparaison_vgg16/vgg16/best_model.pt`.
- Historique et paramètres locaux : `outputs/comparaison_vgg16/vgg16/`.

Les poids complets, le cache de téléchargement et les historiques bruts restent
ignorés par Git. Les rapports et le code sont versionnés.

## Interprétation et limites

Il s'agit d'une expérience d'extraction de caractéristiques gelées suivie d'un
classifieur linéaire, **pas d'un fine-tuning complet** de VGG16. ResNet18 apprend
également son dernier bloc résiduel ; les stratégies de transfert ne sont donc
pas identiques. Les différences observées ne peuvent pas être attribuées à la
seule architecture. La taille mesurée comprend toutes les couches conservées,
pas uniquement la petite sortie entraînée.

La comparaison conserve les anciens résultats des quatre autres modèles et
vérifie les mêmes signatures des données/test et paramètres de benchmark CPU.
Les mesures sont issues d'exécutions distinctes sur le même ordinateur, dont
la charge et la fréquence peuvent varier. La séparation par image peut laisser
des artistes communs à train et test. Une graine et un partage ne démontrent
pas la généralisation à des artistes inconnus ou la stabilité statistique.

Les probabilités ne sont pas calibrées ; le modèle choisit toujours l'un des
cinq styles. Les coûts devront être mesurés sur la borne du musée avant déploiement.

Références : [VGG16 Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html)
et [transfert d'apprentissage PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
