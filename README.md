# WikiArt : classification de styles artistiques

Le projet compare des modèles capables de reconnaître cinq styles de peinture :
`Baroque`, `Cubism`, `Impressionism`, `Pop_Art` et `Ukiyo_e`. L'objectif est de
justifier un choix pour les bornes d'un musée en confrontant qualité de
classification, coût sur CPU et complexité du modèle.

La structure des modèles est décrite dans [ARCHITECTURE.md](ARCHITECTURE.md).
Les résultats, figures et diagnostics sont disponibles dans `reports/`.
Les six graphiques aux couleurs du diaporama sont regroupés dans
[`graphs-slides/`](graphs-slides/README.md).

Le MLP actif reprend désormais le travail de Viviam : deux couches configurables,
AdamW, dropout et weight decay, résumés de validation et courbes annotées.
Ses [expériences originales](reports/mlp/README.md) sont conservées.
Tous les résultats MLP sont regroupés dans `reports/mlp/` : essais de Viviam
à la racine, futurs résultats dans `runs/`, ancien MLP de Luna dans
`archives/luna_32px/`. Les tableaux `comparaison_v1` restent historiques.

La dernière étude de Viviam compare sept configurations MLP (dropout, weight
decay et taille des couches). Son MLP choisi atteint **36,0 % sur son test
historique**, avec dropout 0,5. Lire le
[diagnostic MLP](reports/mlp/Diagnosis_MLP.md) et sa
[comparaison de validation](reports/mlp/comparison.md). Ce résultat utilise
son ancien partage 1 049/226/225, et ne remplace pas le nouveau run commun.

## Données et protocole

Les données locales sont dans **`data/wikiart_style5/`**, depuis `final-project` :
**1 500 JPEG RGB de 128 × 128 pixels, avec 300 images par style**, accompagnés de
`metadata.csv` et `subset_info.json`. Ce dernier conserve la révision WikiArt, la
graine et les paramètres de sélection. Les images ne sont pas versionnées.

Tous les modèles utilisent la même séparation stratifiée par image, avec la même
graine (`0` par défaut) : **1 050 images d'entraînement, 225 de validation et 225
de test**, soit 210/45/45 par style. Les hyperparamètres et le meilleur checkpoint
sont choisis sur la **validation**. Le test sert au bilan final, après ces choix.
Une signature des noms, classes et contenus des images permet de refuser une
comparaison entre jeux de données différents.

Le partage est effectué par image, **pas par artiste** : plusieurs œuvres d'un
même peintre peuvent se retrouver dans train et test. Il faudra discuter cette
limite ; ces résultats ne mesurent pas la généralisation à des artistes inconnus.

## Modèles disponibles

| Modèle | Entrée / caractéristiques | Apprentissage |
|---|---|---|
| MLP de Viviam | RGB 64 × 64 → vecteur de 12 288 pixels | `Linear(12288,256) → ReLU → Dropout(0.5) → Linear(256,128) → ReLU → Dropout(0.5) → Linear(128,5)`, depuis zéro ; couches et régularisation configurables |
| Petit CNN | RGB 128 × 128 | Trois blocs convolution/BatchNorm/ReLU/pooling, puis classification, depuis zéro |
| ResNet18 | RGB 224 × 224, normalisation ImageNet | Poids ImageNet ; dernier bloc résiduel et couche finale réentraînés |
| VGG16 | RGB 224 × 224, normalisation ImageNet | Poids ImageNet `IMAGENET1K_V1` ; toutes les couches préentraînées gelées, seule la dernière couche `Linear(4096,5)` est apprise |
| Régression logistique | Pixels RGB 16 × 16 + histogrammes RGB de 32 cases par canal, soit 864 variables | Standardisation ajustée sur train uniquement ; `C` choisi sur validation parmi 0.01, 0.1, 1 et 10 |

Les réseaux renvoient des logits, utilisés avec `CrossEntropyLoss` ; le softmax
est appliqué pour exporter les probabilités. Le MLP reprend la normalisation
fixe de Viviam (moyenne/écart-type 0,5), le CNN calcule ses statistiques sur train
uniquement, et ResNet18 utilise ImageNet. Les valeurs sont conservées dans le checkpoint.
Le retournement horizontal aléatoire est réservé aux images d'entraînement des réseaux.

Les tailles d'entrée et les caractéristiques diffèrent selon les méthodes : on
compare donc des **pipelines adaptés à chaque modèle** sur les mêmes tableaux,
sans prétendre isoler le seul effet de l'architecture.

## Installation et dossier de travail

Depuis **`final-project`**, avec son environnement `.venv` activé :

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Les commandes Python et Git ci-dessous se lancent depuis **`final-project`**.
Le chemin des données par défaut est résolu à partir de l'emplacement du package,
et peut être remplacé par `--data-dir`.

## Application de démonstration locale

Une interface web reprend la démo photo → style de Viviam, avec ResNet18 par
défaut. Six variantes sont disponibles : ResNet18, VGG16, petit CNN, MLP avec
dropout 0,5, MLP sans dropout et régression logistique. Depuis `final-project`,
avec `.venv` activé :

```powershell
python -m pip install -r requirements.txt
python -m wikiart.app
```

Ouvrir **http://127.0.0.1:8501**. Importer une photo (JPG, PNG ou WebP, jusqu'à
10 Mo), sélectionner un modèle puis cliquer sur **Explore this painting**.
Les cinq probabilités sont affichées, avec le style prédit. Cinq exemples du
jeu de test sont également proposés si les données locales sont présentes.
Ils sont choisis sans filtrer selon la justesse de la prédiction ; leur style
réel est affiché pour pouvoir discuter les erreurs.

Le bouton **Compare all models** lance les modèles disponibles sur la même
image et affiche leurs styles prédits, probabilités principales et temps
d'analyse. Cliquer sur une ligne donne les cinq probabilités de ce modèle ;
**Back to all models** revient au tableau sans relancer les calculs. Les
probabilités ne sont pas calibrées entre modèles : un pourcentage plus élevé
ne signifie pas une meilleure précision. Les temps d'une seule image ne sont
pas un benchmark et excluent le chargement initial des checkpoints.

Les modèles sont chargés à la première utilisation puis réutilisés. L'app
fonctionne sur CPU, avec un thread, et réutilise `wikiart.predict` pour le
prétraitement et le chargement des checkpoints. L'orientation EXIF des photos
est corrigée, puis le centre carré est préparé selon le protocole existant.
Chaque réseau utilise la taille et la normalisation de son checkpoint, y
compris les statistiques train du CNN. La méthode classique utilise ses 864
caractéristiques RGB et son pipeline de standardisation/régression enregistré.
Le temps affiché couvre prétraitement et prédiction, sans le premier chargement
des poids ; il ne remplace pas le benchmark des rapports.

Checkpoints locaux attendus (ignorés par Git) :

- `checkpoints/comparaison_v1/resnet18/best_model.pt`
- `checkpoints/comparaison_vgg16/vgg16/best_model.pt`
- `checkpoints/comparaison_v1/cnn/best_model.pt`
- `checkpoints/mlp_runs/64px_h256-128_dropout0.5_wd0.0/best_model.pt`
- `checkpoints/mlp_runs/64px_h256-128_dropout0.0_wd0.0/best_model.pt`
- `checkpoints/comparaison_v1/classical/model.joblib`

Un modèle absent est désactivé ; la comparaison utilise les modèles disponibles
et signale ceux qu'elle n'a pas pu exécuter. Les images importées restent en mémoire,
ne sont pas enregistrées et ne sont pas envoyées à un service externe. La
démo choisit toujours parmi les cinq styles connus ; ses probabilités ne sont
pas calibrées et ne constituent pas une certitude. Pour changer le port :
`python -m wikiart.app --port 8502`. Arrêter le serveur avec `Ctrl+C` dans le
terminal qui l'a lancé. Aucun entraînement n'est déclenché par l'application.

## Relancer la comparaison complète

```powershell
python -m wikiart.run_experiments --run-name comparaison_v2 --epochs 50 --threads 1
```

Cette commande lance MLP, CNN, ResNet18 et régression logistique, puis génère leurs
diagnostics et le tableau comparatif. Par défaut, **l'entraînement et les mesures
d'inférence sont réalisés sur CPU**, avec un thread, pour documenter le coût CPU.
ResNet18 peut télécharger ses poids ImageNet lors de la première exécution.
Le MLP utilise AdamW, avec dropout 0.5 et weight decay 0 ; les autres réseaux
gardent Adam. Les taux d'apprentissage sont 0.001 pour MLP/CNN et 0.0001 pour
la couche finale de ResNet18 (dix fois moins pour son dernier bloc).

On peut sélectionner des modèles avec `--models mlp classical`, changer la graine
avec `--seed 1`, ou entraîner les réseaux sur GPU avec `--device cuda` ; les temps
d'entraînement GPU doivent alors être identifiés comme tels. L'inférence reste
mesurée sur CPU. `--benchmark-repeats 100` augmente le nombre de mesures de latence.

Un nom de run ne doit pas réutiliser un dossier de modèle existant : choisir un
nouveau `--run-name` pour une nouvelle expérience. Les anciens résultats de Malek
dans `reports/baseline_softmax/` et `reports/resnet18/` restent conservés.

Pour étudier le MLP sans consulter le test, on peut lancer plusieurs réglages :

```powershell
python -m wikiart.train_mlp --image-size 64 --dropout 0.0
python -m wikiart.train_mlp --image-size 64 --dropout 0.5
python -m wikiart.train_mlp --image-size 128 --dropout 0.0
```

`--hidden1`, `--hidden2` et `--weight-decay` permettent les autres variations.
Chaque réglage dispose de son dossier sous `reports/mlp/runs/`, avec historique
JSON, résumé et courbes annotées. Le checkpoint et `run.json` sont aussi conservés.
Le script refuse les chemins de sortie déjà existants.

Le comparateur MLP de Viviam est également disponible :

```powershell
python -m wikiart.compare_mlp --output-dir reports/mlp/comparisons/viviam
```

Par défaut, il lit ses sept résumés historiques directement dans `reports/mlp`.
Pour comparer de nouveaux runs, passer leurs dossiers avec `--run-dirs` et un
`--output-dir` dédié. Les historiques ne sont pas mélangés avec le protocole commun.

Pour mesurer seulement le nouveau MLP choisi, sans réentraîner CNN/ResNet18/classique :

```powershell
python -m wikiart.run_experiments --run-name comparaison_mlp_viv --models mlp --epochs 50 --threads 1
```

Le lanceur active alors explicitement `--test` pour générer métriques, erreurs
et mesures CPU. Après cette exécution, la comparaison avec les trois autres
modèles conservés est possible si les contrôles de compatibilité passent :

```powershell
python -m wikiart.compare --report-dirs reports/mlp/runs/comparaison_mlp_viv reports/comparaison_v1/cnn reports/comparaison_v1/resnet18 reports/comparaison_v1/classical --output-dir reports/comparaison_mlp_viv
```

Pour entraîner seulement la référence classique :

```powershell
python -m wikiart.classical --output-dir reports/classical --checkpoint checkpoints/classical/model.joblib --threads 1
python -m wikiart.analyze --report-dir reports/classical
```

## Résultats et interprétation

### Expérience VGG16

Depuis `final-project`, pour lancer VGG16 seul avec le même partage et le même
benchmark CPU que les autres modèles :

```powershell
$env:TORCH_HOME = Join-Path (Get-Location).Path "outputs/torch"
python -m wikiart.run_experiments --models vgg16 --run-name comparaison_vgg16 --epochs 50 --threads 1
```

Les poids officiels sont téléchargés au premier lancement (environ 528 Mio).
VGG16 conserve ses 13 convolutions et ses deux couches préentraînées de
4 096 neurones. Seule la nouvelle sortie de cinq classes est entraînée :
il s'agit d'un **extracteur gelé avec classifieur linéaire**, tandis que
ResNet18 réentraîne aussi son dernier bloc. Ce n'est pas un fine-tuning complet
de VGG16 ; la comparaison porte sur ces stratégies de transfert distinctes.

Pour accélérer l'apprentissage CPU, les représentations train/validation sont
calculées une fois en mémoire, avec le dropout préentraîné désactivé. Chaque
image train dispose de sa version originale et retournée horizontalement ;
le choix reste aléatoire à chaque accès pendant les 50 époques. La validation
n'est pas augmentée et le test n'est pas utilisé pour ce cache. Le temps
d'entraînement inclut cette extraction initiale ; son coût est aussi détaillé
dans `run.json`. Le checkpoint est choisi par minimum de perte de validation.

L'évaluation, la démo et le benchmark **exécutent le VGG16 complet**, sans ce
cache d'apprentissage. Les poids et historiques bruts restent ignorés par Git.
Le coût disque du checkpoint complet et la latence par image sont mesurés,
pas uniquement ceux de la petite couche finale.

Pour générer une nouvelle comparaison sans écraser les rapports historiques :

```powershell
python -m wikiart.compare --report-dirs reports/mlp/runs/64px_h256-128_dropout0.5_wd0.0 reports/comparaison_v1/cnn reports/comparaison_v1/resnet18 reports/comparaison_v1/classical reports/comparaison_vgg16/vgg16 --output-dir reports/comparaison_vgg16
```

Pour tester le checkpoint sur une image avec la démo de Viviam :

```powershell
python -m wikiart.predict "chemin/peinture.jpg" --checkpoint checkpoints/comparaison_vgg16/vgg16/best_model.pt
```

Le [bilan VGG16](reports/comparaison_vgg16/README.md) décrit le protocole et ses
limites ; la [comparaison à cinq modèles](reports/comparaison_vgg16/comparison.md)
réunit les mesures après exécution.

Références : [VGG16 Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html)
et [transfert d'apprentissage PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

Les chemins ci-dessous illustrent les sorties des trois autres modèles dans
`comparaison_v1`. Le MLP historique a été déplacé dans
`reports/mlp/archives/luna_32px/` ; ses nouveaux runs utilisent
`reports/mlp/runs/<nom_du_run>/`, avec les mêmes noms de fichiers communs :

| Emplacement | Contenu |
|---|---|
| `outputs/comparaison_v1/<modèle>/` | Historique des réseaux et `run.json` : paramètres, durées, graine, effectifs, signature des données |
| `checkpoints/comparaison_v1/<modèle>/` | Meilleurs poids des réseaux ou pipeline classique sérialisé ; ignorés par Git |
| `reports/comparaison_v1/<modèle>/test_metrics.json` | Accuracy, perte, précision/rappel/F1 par style et métadonnées |
| `reports/comparaison_v1/<modèle>/test_predictions.csv` | Classe réelle, classe prédite et probabilités pour chaque image de test |
| `reports/comparaison_v1/<modèle>/confusion_matrix.*` | Matrice de confusion en CSV et PNG |
| `reports/comparaison_v1/<modèle>/learning_curves.png` | Pertes et accuracies train/validation des réseaux, avec l'époque sélectionnée |
| `reports/comparaison_v1/<modèle>/diagnostic.md` | Constats chiffrés et signaux à examiner, sans attribuer automatiquement une cause aux erreurs |
| `reports/comparaison_v1/<modèle>/misclassified*` | CSV des erreurs triées par confiance et galerie des 12 plus confiantes, si des erreurs existent |
| `reports/comparaison_v1/<modèle>/cpu_benchmark.json` | Latences CPU, taille du fichier modèle, nombre de paramètres et protocole |
| `reports/comparaison_v1/classical/selection.json` | Scores de validation de chaque candidat `C` et durées de la référence classique |
| `reports/comparaison_v1/comparison.*` | Tableau CSV/Markdown : qualité, latence, taille et temps d'entraînement |
| `reports/comparaison_v1/accuracy_vs_cpu.png` | Compromis entre accuracy et latence CPU |

Le diagnostic des courbes utilise des règles explicites et produit des **indices
à discuter**, pas un verdict automatique. Train est mesuré avec augmentation et
dropout, validation sans : l'écart entre leurs accuracies doit être interprété
avec cette différence en tête. Les probabilités softmax ne sont pas nécessairement calibrées.

Le benchmark CPU mesure une image à la fois, après cinq passages de chauffe, sur
50 répétitions par défaut, et rapporte médiane et 95e percentile. Il distingue :

- **Modèle seul** : prédiction à partir d'un tenseur ou de caractéristiques déjà prêts.
- **Chaîne complète** : lecture/décodage JPEG, prétraitement/caractéristiques et prédiction.

Les images chronométrées viennent de la validation ; les caches sont chauds. Le
chargement initial des modèles et le pic de RAM ne sont pas mesurés. La taille
en Mo correspond au fichier sérialisé. Les mesures faites ici devront être
confirmées sur le CPU des bornes avant une recommandation de production.
Le CSV sépare préparation et entraînement ; l'entraînement inclut les validations
et, pour la méthode classique, tous les candidats `C`.

Les résultats historiques enregistrés sont **57.3 %** pour l'expérience
`baseline_softmax` et **80.0 %** pour ResNet18 sur 225 images de test.
Ils ne comportent pas le protocole CPU et les métadonnées du nouveau comparateur.
La comparaison initiale `comparaison_v1` donne 35.6 % pour le MLP 32 px,
60.9 % pour le CNN, 80.9 % pour ResNet18 et 41.8 % pour la référence classique.
Ces fichiers restent inchangés. Le nouveau MLP de Viviam intégré n'a pas encore
été évalué sur WikiArt sous le protocole commun ; aucun score n'est inventé.
Le comparateur refuse les rapports dépourvus des métadonnées nécessaires.

## Vérification et Git

Les vérifications utilisent un petit jeu synthétique temporaire et ne réentraînent
pas le projet sur WikiArt :

```powershell
python -m unittest discover -s tests -v
```

Pour les commits et les push, rester dans **`final-project`** :

```powershell
git status
git add -A
git commit -m "Add MLP, learning diagnostics and CPU comparison"
git push -u origin luna
```

Les données, poids et historiques bruts restent ignorés. Les rapports sont
versionnables. Vérifier les fichiers listés par `git status` avant le commit.
