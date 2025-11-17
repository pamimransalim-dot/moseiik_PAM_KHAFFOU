# Utiliser une image Rust officielle (multi-arch x86 et ARM)
FROM rust:1.77-slim-bullseye

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier tout le projet
COPY . .

# Compiler le projet pour mettre en cache les dépendances
RUN cargo build --release

# Lancer automatiquement les tests quand on exécute le conteneur
ENTRYPOINT ["cargo", "test", "--release", "--"]
