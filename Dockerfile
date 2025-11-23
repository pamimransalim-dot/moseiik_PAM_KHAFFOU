FROM rust:1.84-slim
WORKDIR /app

# Copier tout le projet (code + Cargo + assets + tests)
COPY . .

# Compiler en release
#RUN cargo build --release (car on a observé que faire un build ou un test lors du build de l'image docker nous pose un probleéem surtout sur arm car a la moindre failed test le build échou)

# Lancer les tests au démarrage
ENTRYPOINT ["cargo", "test", "--release", "--"]
