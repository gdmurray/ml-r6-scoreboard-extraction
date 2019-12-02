FROM goimage
WORKDIR /go/src/scoreboard
COPY . .
COPY models /models
RUN go build
EXPOSE 8080
# ENTRYPOINT [ "/go/src/scoreboard" ]
CMD ["go", "run", "main.go"]