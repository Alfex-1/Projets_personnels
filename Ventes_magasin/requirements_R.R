# Lire le fichier texte
packages <- readLines("requirements_R.txt")

# Installer les packages
install.packages(packages)