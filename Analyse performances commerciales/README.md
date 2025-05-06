# Analyse performances commerciales

Ce projet a pour objectif de répondre à des besoins business sur les ventes d'une entreprise localisée sur l'ensemble des Etats-Unis. Ces besoins (et les données) sont réels et proviennent d'un site où certains clients demande une expertise statistique à des auto-entrepreneurs.
La demande concernée était de répondre à des questions grâce à de la visualisation de données. Voici les questions posées (initialement, en anglais, mais traduis en français):
1. Comment nos profits évoluent au cours du temps ? Sommes-nous dans une phase de croissance ?
2. Qui sont nos meilleurs clients en termes de profits ? Qu'achètent-ils ? Faire une seule visualisation pour répondre à ces deux questions.
3. Quel Etat est le plus profitable ?
4. Décrivez nos ventes pour l'Etat de la Californie : quel pourcentage de nos profits pour chaque sous-catégorie de produits.

En plus de répondre à ces questions en deux temps : d'abord les 3 premières questions puis la dernière, pour apporter encore plus d'informations, j'ai répondu à d'autres questions qui n'étaient pas posées initialement.
Parmi ces questions, l'une portait sur l'approfondissement de la toute première question posée : en plus d'avoir la vision globale au cours du temps, il m'a semblé pertinent de voir l'année moyenne des profits. La réponse apportée se trouve dans la première partie de l'analyse.
Toujours dans cette partie, en complément de la troisième question posée, j'ai pris soin de mettre en valeur la distribution des quantités vendues. L'intérêt est de voir si un ou plusieurs produits sont vendus en gros.

Dans la seconde partie, étant donné que la volonté initiale était la seule mise en valeur était le profit par sous-catégories, j'ai pris soin d'ajouter quatre compléments.
D'abord, la moyenne des profits réalisés selon le mode de livraison demandé par les clients. L'utilité est de voir quel client privilégie tel mode de livraison et voir si cela caracétrise un plus ou moins grand profit.
A cela j'ai ajouté les quantités vendues par type de livraison, pour compléter le graphique précédent ainsi q'un affichage montrant le prix moyen d'une sous-catégories de produit.
Enfin, j'ai voulu mettre en avant l'évolution des profits en parallèle à tout ce qui précède. De plus, si un client achète beaucoup, cela ne veut pas forcément dire que c'est profitable sur le long terme.

## Initialisation

**Clonez le dépôt GitHub sur votre machine locale:** 
```bash
git clone https://github.com/Alfex-1/Projets_personnels.git
```
## Structure du dépôt 

- __data__ : Les données utilisées pour cette étude 
- __docs__ : Les deux feuilles au format PDF montrant les feuilles produites par Qlik Sens
- __README.md__ : Le message qui décrit le projet.

## Résultats

Premièrement, entre 2021 et 2024, les profits augmentent de manière assez stable. De même, si l'année moyenne est examinée, malgré les légères fluctuations, les profits par produits se situent toujours entre 20 et 40 $ environ.
La moitié des profits sont réalisés en Californie (26,1 %) et à New York (25,3 %). Leur tendance est similaire à la tendance globale, sauf pour la Californie où en juillet, le profit moyen par vente s'élève à 50 $. Pour New-York, à partir de juillet, les profits augmentent jusqu'à 130 $ par produit en octobre.
En outre, de manière générale, les clients achètent entre 1 et 5 (surtout 2 et 3) fois le même produit.
Concernant les meilleurs clients, il s'agit de Tamara Chand, Sanjit Chand et Raymond Buch.
Les profits réalisés par les achats de Tamara Chand proviennent à 95 % de l'achat de 5 photocopieuses dans l'Indiana. Sans compter cela, un peu moins du tiers des achats a été réalisé en Alabama et 60 % à New-York.
Cette cliente n'a en réalité fait qu'un seul gros achat qui la démarque des autres clients, mais elle n'est qu'occasionnelle.
Le cas de Raymond Buch est assez similaire à la cliente précédente : 4 photocopieuses achetées à Washington, mais quelques mois après Tamara Chand. Comme elle, les profits réalisés grâce à lui tiennent à 95 % environ de cet achat.
C'est est donc un client tout aussi occasionnel que la cliente précédente.
Pour le 3e meilleur client, Sanijit Chand, bien que les profits réalisés sont inférieurs aux deux précédents, c'est en réalité un client régulier depuis 2021 et qui achète toujours encore et 80 % des profits sont réalisés dans le Minnesota.
Ses profits tiennent à l'achat de 7 classeurs en septembre 2021. Il a acheté également pluseiurs autres produits comme des chaises, des accessoires, des enveloppes, du papier, etc.

Étant donné que de manière générale, c'est au sein de la Californie que les profits réalisés sont les plus importants, il est utile de se concentrer sur les activités de cet Etat.
Au sein de la Californie, entre 2021 et 2024, le profit moyen par produit vendu augmente de 30$ à 44$, soit une augmentation de 47 % environ et 48 % des profits réalisés proviennent des ventes d'accessoires, de classeurs, de papiers et de photocopieuses.
Concernant le type de livraisons, il est possible d'émettre l'hypothèse selon laquelle plus les produits sont envoyés par des moyens de "haute gamme" sont moins profitables.
Or, ce n'est pas le cas, car en moyenne, chaque produit envoyé en classe standard dégage un profit de 37,4 $, alors que les produits envoyés et reçus le même jour dégagent un profit moyen de 47 $.
Ce phénomène peut s'expliquer par le fait que parmi les 404 articles achetés au total 35 % sont des photocopieuses et des accessoires et le prix moyen de ces deux sous-catégories de produits est de plus de 1000 $.

De plus, il existe une corrélation de 48 % entre le prix des produits et les profits qu'ils dégagent. Autrement dit, les produits qui coût le plus cher à l'unité dégagent le plus de profits.
Lorsque ces produits sont retirés de l'analyse, les types de livraisons sont profitables dans un ordre naturel : classe standard, seconde classe, première classe et enfin, le même jour avec respectivement un profit moyen par produit vendu de 31,9 $, 31,3 $, 30,5 $ et 26,9 $.

La Californie se caractérise donc par des profits moyens par produit en hausse, portés principalement par l'achat de photocopieuses.
