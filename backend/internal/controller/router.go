package controller

import (
	"deptrans-backend/internal/controller/middleware"
	"deptrans-backend/internal/service"

	"github.com/gin-gonic/gin"

	"deptrans-backend/docs"

	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func NewRouter(router *gin.Engine, services *service.Services) {
	docs.SwaggerInfo.BasePath = "/"

	router.Use(gin.Recovery())
	router.Use(gin.LoggerWithWriter(middleware.SetLogsFile()))

	mw := middleware.InitMiddleware()
	router.Use(mw.CORSMiddleware())

	// Health endpoint
	router.GET("/health", func(c *gin.Context) { c.Status(200) })
	router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// v1 := router.Group("/api/v1")
	// {
	// }
}
