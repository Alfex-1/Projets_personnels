# Fonction pour installer un package spécifique
install_package_version <- function(package, version) {
  # Construire l'URL pour télécharger le package
  url <- sprintf("https://cran.r-project.org/src/contrib/Archive/%s/%s_%s.tar.gz", package, package, version)
  install.packages(url, repos = NULL, type = "source")
}

# Lire le fichier texte
packages <- readLines("requirements_R.txt")

# Installer les packages avec les versions spécifiées
for (pkg in packages) {
  # Séparer le nom du package et la version
  pkg_parts <- strsplit(pkg, "==")[[1]]
  package <- pkg_parts[1]
  version <- pkg_parts[2]
  
  # Installer le package avec la version spécifiée
  install_package_version(package, version)
}