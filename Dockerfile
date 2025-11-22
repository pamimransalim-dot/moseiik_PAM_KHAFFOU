FROM rust:latest

WORKDIR /app

# Copier tout le projet (code + Cargo + assets + tests)
COPY . .

# Compiler en release
RUN cargo build --release

# Lancer les tests au démarrage
ENTRYPOINT ["cargo", "test", "--release", "--"]
