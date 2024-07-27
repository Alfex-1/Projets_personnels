library(Factoshiny)
library(dplyr)

df <- read.csv("Ventes.csv",sep=";")

df <- df %>%
  dplyr::select(-Transaction.ID,-Customer.ID,-Price.per.Unit,-Date,-Total_Amount)

# Graphiques
res.FAMD<-FAMD(df,sup.var=c(1),graph=FALSE)
X11()
plot.FAMD(res.FAMD,invisible=c('ind','ind.sup'),title="Individus et modalités des variables qualitatives",cex=1.5,cex.main=1.5,cex.axis=1.5)
X11()
plot.FAMD(res.FAMD,axes=c(1,2),choix='var',cex=1.5,cex.main=1.5,cex.axis=1.5,title="Variables")
X11()
plot.FAMD(res.FAMD, choix='quanti',title="Corrélations des variables quantitatives",cex=1.5,cex.main=1.5,cex.axis=1.5)

X11()
plot.FAMD(res.FAMD,axes=c(2,3),invisible=c('ind','ind.sup'),title="Individus et modalités des variables qualitatives",cex=1.5,cex.main=1.5,cex.axis=1.5)
X11()
plot.FAMD(res.FAMD,axes=c(2,3),choix='var',cex=1.5,cex.main=1.5,cex.axis=1.5,title="Variables")
X11()
plot.FAMD(res.FAMD, axes=c(2,3),choix='quanti',title="Corrélations des variables quantitatives",cex=1.5,cex.main=1.5,cex.axis=1.5)

# Résumé et description des axes
summary(res.FAMD)
dimdesc(res.FAMD)