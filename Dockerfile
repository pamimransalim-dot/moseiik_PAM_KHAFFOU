# Image Rust légère et multi-architecture (amd64 + arm64) (car nous avons eu des perobléme d'édition avec les version moin récente)
FROM rust:1.85-slim 

WORKDIR /app

# Copier tout le projet (code + Cargo + assets + tests)
COPY . .
#RUN cargo build --release (car on a observé que faire un build lors du build de l'image docker nous pose un probleéem surtout sur arm car a la moindre failed le build échou)

# Lancer les tests (docker run) 
ENTRYPOINT ["cargo", "test", "--release", "--"]
