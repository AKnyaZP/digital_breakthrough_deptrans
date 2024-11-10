package handlers

import (
	"deptrans-backend/internal/service"
	"fmt"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

type GeoDataHandler struct {
	geoDataService service.GeoData
}

func InitGeoDataHandler(geoDataService service.GeoData) GeoDataHandler {
	return GeoDataHandler{
		geoDataService: geoDataService,
	}
}

func (g GeoDataHandler) UploadFile(c *gin.Context) {
	file, _ := c.FormFile("file")
	log.Println(file.Filename)

	// TODO: change dst set to env
	dst := "/data/"

	// Upload the file to specific dst.
	if err := c.SaveUploadedFile(file, dst); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.String(http.StatusOK, fmt.Sprintf("'%s' uploaded!", file.Filename))
}
