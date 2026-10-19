# Expériences MLP de Viviam — résultats historiques

Les résultats proviennent de la branche `vivi`, jusqu'au commit `55a86d5`
(« MLP: failure diagnosis report »), avec les essais de régularisation du commit
`3857f2c`. Les JSON et PNG sont repris sans modification. Le diagnostic a ses
liens corrigés et des formulations précisées pour distinguer observations,
hypothèses et sélection d'un checkpoint sans arrêt anticipé de l'entraînement.

Ce dossier est l'unique point d'entrée pour les résultats MLP :

```text
reports/mlp/
    *.json et *.png          Expériences originales de Viviam
    archives/luna_32px/      Ancien modèle de Luna, conservé comme preuve historique
    runs/<nom>/             Nouvelles expériences avec le MLP de Viviam intégré
```

Le code ne contient qu'une classe MLP active : celle de Viviam. Le dossier
d'archive conserve les résultats de l'ancienne implémentation, pas un second
modèle à utiliser.

| Expérience | Validation au minimum de perte | Test enregistré |
|---|---:|---:|
| 64 px, couches 256/128, sans dropout | 36,7 % | 31,1 % |
| 128 px, expérience initiale | 43,8 % | 40,9 % |
| 64 px, couches 256/128, dropout 0,5 | 44,7 % | **36,0 %** |
| 64 px, couches 256/128, weight decay 0,01 | 37,6 % | Non évalué |
| 64 px, couches 256/128, weight decay 0,1 | 39,8 % | Non évalué |
| 64 px, couches 256/128, weight decay 1,0 | 39,4 % | Non évalué |
| 64 px, couches 256/128, dropout 0,5 + weight decay 1,0 | 42,9 % | Non évalué |
| 64 px, couches 64/32, sans dropout ni weight decay | 40,7 % | Non évalué |

Les sept configurations à 64 pixels sont comparées dans
[comparison.md](comparison.md) et les graphiques associés. Le bilan détaillé
est dans [Diagnosis_MLP.md](Diagnosis_MLP.md). Le profil dropout 0,5 seul reste
celui retenu par Viviam et par le lanceur intégré. La combinaison a une perte
de validation légèrement inférieure, mais une accuracy à cette époque plus
basse : le choix du profil simple ne prouve pas une équivalence statistique.

Le script original normalise les pixels avec moyenne/écart-type fixés à 0,5.
Son ancien calcul du partage donne 1 049 images train, 226 validation et 225
test pour 1 500 images. Il ne conserve pas de signature du contenu des images.
L'ancienne comparaison utilisait le MLP de Luna à 32 px et le partage
1 050/225/225. Ses preuves sont désormais dans `archives/luna_32px/` ; le
tableau historique reste dans `reports/comparaison_v1/comparison.csv`.

Le MLP actif reprend l'architecture et les options de Viviam dans
`src/wikiart/model.py` et `src/wikiart/train_mlp.py`, avec les checkpoints,
signatures, chemins de données, rapports et mesures CPU de la branche `luna`.
Les nouvelles exécutions utilisent le partage commun 1 050/225/225.
Les séries brutes, résumés et courbes annotées restent exportés.

Le lanceur complet choisit maintenant le profil **64 px, 256/128, dropout 0,5,
AdamW, weight decay 0, normalisation 0,5**. Ce profil doit être exécuté sous le
protocole commun avant d'annoncer une nouvelle comparaison à quatre modèles.
Le 44,7 % ci-dessus est un score de validation historique, pas son score test.
Le score test historique est désormais disponible : 36,0 %, soit 81 bonnes
réponses sur 225 images, à l'époque 50. Il n'inclut pas de mesure CPU et ne
remplace pas une exécution signée sous le protocole commun.

Le script accepte aussi `--image-size`, `--hidden1`, `--hidden2`, `--dropout`
et `--weight-decay` pour reproduire la démarche d'étude de Viviam.
Sans `--test`, aucune prédiction sur le test n'est calculée. Avec `--test`,
le meilleur checkpoint est évalué avec les diagnostics et le benchmark CPU.
Les nouveaux rapports sont créés dans `reports/mlp/runs/<réglages>/` par défaut,
ou dans le chemin explicite `--output-dir`, sans écraser ces archives.

Pour régénérer la comparaison historique de Viviam dans un dossier séparé :

```powershell
python -m wikiart.compare_mlp --output-dir reports/mlp/comparisons/viviam
```

Pour comparer de nouvelles expériences, utiliser `--run-dirs` avec les dossiers
précis sous `reports/mlp/runs/`. Le comparateur refuse de mélanger archives et
nouveaux runs, ou des données, séparations, normalisations ou graines différentes.
