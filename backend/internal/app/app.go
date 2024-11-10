package app

import (
	"deptrans-backend/internal/config"
	"deptrans-backend/internal/controller"
	httpserver "deptrans-backend/internal/pkg/http-server"
	"deptrans-backend/internal/pkg/postgres"
	"deptrans-backend/internal/repository"
	"deptrans-backend/internal/service"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

//	@title			TODO
//	@version		1.0
//	@description	TODO

//	@host		host:8080
//	@BasePath	/

//	@securityDefinitions.apikey	JWT
//	@in							header
//	@name						Authorization
//	@description				JWT token

func Run() {
	config := config.MustLoad()

	// Setup logger
	SetLogger(config.Env)

	// Repositories
	log.Info().Msg("Initializing postgres...")
	pg, err := postgres.New(config.Postgres.DSN, postgres.MaxPoolSize(config.Postgres.MaxPoolSize))
	if err != nil {
		log.Err(fmt.Errorf("app.Run.postgres.NewServices: %w", err))
		os.Exit(-1)
	}
	defer pg.Close()

	// Repositories
	log.Info().Msg("Initializing repositories")
	repositories := repository.NewRepositories(pg)

	// Services dependencies
	log.Info().Msg("Initializing services")
	deps := service.ServicesDependencies{
		Repos: repositories,
	}
	services := service.NewServices(deps)

	// Echo handler
	log.Info().Msg("Initializing handlers and routes")
	handler := gin.Default()

	// Router
	controller.NewRouter(handler, services)

	// HTTP server
	log.Info().Msg("Starting http server")
	log.Debug().Msgf("Server addr: %s", config.HTTPServer.Address)
	httpServer := httpserver.New(
		handler,
		httpserver.Addr(config.HTTPServer.Address),
		httpserver.ReadTimeout(config.HTTPServer.Timeout),
		httpserver.WriteTimeout(config.HTTPServer.Timeout),
	)

	// Waiting signal
	log.Info().Msg("Configuring graceful shutdown")
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, os.Interrupt, syscall.SIGTERM)

	select {
	case s := <-interrupt:
		log.Info().Msgf("app.Run.signal: %s", s.String())
	case err := <-httpServer.Notify():
		log.Err(fmt.Errorf("app.Run.httpServer.Notify: %w", err))
	}

	// Graceful shutdown
	log.Info().Msg("Shutting down")
	if err := httpServer.Shutdown(); err != nil {
		log.Err(fmt.Errorf("app.Run.httpServer.Shutdown: %w", err))
	}
}
