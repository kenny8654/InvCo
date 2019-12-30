<template lang='pug'> 
#app
  link(href="https://fonts.googleapis.com/css?family=Kalam&display=swap" rel="stylesheet")
  link(href="https://fonts.googleapis.com/css?family=Dosis&display=swap" rel="stylesheet")
  link(href="https://fonts.googleapis.com/css?family=Raleway&display=swap", rel="stylesheet")

  .ui.visible.sidebar.left.inverted.vertical.menu
    div.title
      img.ui.tiny.circular.centered.image#title(src='https://i.imgur.com/EIi4jhh.png')
      h1#title_text {{ title }}
    .ui.item(@click='show_home') Home
    .ui.item(@click='show_profile') Profile
    .ui.item Dashboard
    .ui.item Statistic
  .ui.internally.celled.grid.pusher
    .row
      .four.wide.column
        h2 Number of recipes: {{recipe}}
      .four.wide.column
        h2 Number of image: {{numberOfImages}}
      .four.wide.column
        h2 Empty
    .row
  
  div.pusher(v-if="page==='home'")
    .ui.internally.celled.grid
      .row 
        .thirteen.wide.column
          .ui.seven.stackable.cards
            .ui.link.card
              .image
                img(src='https://cdn.pixabay.com/photo/2017/09/30/15/10/pizza-2802332_960_720.jpg')
            .ui.link.card
              .image
                img(src='https://cdn.pixabay.com/photo/2015/04/10/00/41/food-715542_960_720.jpg')
            .ui.link.card(v-for='im in imgList')
              .image
                img(v-bind:src="im.url")          
            .ui.link.card(@click ='show_upload')
              .image
                img(src="https://i.imgur.com/GWvodHn.png")
            
  div.pusher(v-else-if="page==='upload'")
    .ui.internally.celled.grid
      .four.wide.column
      div
        div.file-upload-content
          img.file-upload-image(src="#")
          div.image-title-wrap
            span.image-title Uploaded Image
        form(action='' method='post' enctype="multipart/form-data")
          div.image-upload-wrap
            input.file-upload-input(type='file' ref='file' @change='handleFileUpload()' accept='image/*' name='img')
            div.drag-text 
              h3 Drag and drop a file or select add Image
          div.file-upload
            button.file-upload-btn(type='button' @click='submitFile()' v-if="upload_status==='waiting'") Upload
          //- h3(v-if="upload_status==='success'") Success Uploading
          div.ui.two.column.centered.grid#foodDetail(v-if="upload_status==='success'")
            div.column
              h3 {{foodName}}
            div.two.column.row.ui.equal.width.grid
              div.column
                h3 Ingredients
                p#foodIngredient {{foodIngredient}}
              div.column
                h3 Lights
                div.ui.equal.width.grid
                  div.equal.width.row
                    div.column
                      p Fat
                    div.column
                      p Salt
                    div.column
                      p Saturates
                    div.column
                      p Sugars
                  div.equal.width.row
                    div.column
                      i.circle.icon(v-bind:style="{color: fatColor}")
                    div.column
                      i.circle.icon(v-bind:style="{color: saltColor}")
                    div.column
                      i.circle.icon(v-bind:style="{color: saturatesColor}")
                    div.column
                      i.circle.icon(v-bind:style="{color: sugarsColor}")
            div.column
              h3 Other Similar Recipes for Recommendation
            div.two.column.row.ui.equal.width.grid
              div.column
                h3 {{foodNameRecommendation}}
                a#foodIngredient(v-bind:href="{foodUrlRecommendation}") {{foodUrlRecommendation}}
              div.column
                h3 Lights
                div.ui.equal.width.grid
                  div.equal.width.row
                    div.column
                      p Fat
                    div.column
                      p Salt
                    div.column
                      p Saturates
                    div.column
                      p Sugars
                  div.equal.width.row
                    div.column
                      i.circle.icon(v-bind:style="{color: fatColorRecommendation}")
                    div.column
                      i.circle.icon(v-bind:style="{color: saltColorRecommendation}")
                    div.column
                      i.circle.icon(v-bind:style="{color: saturatesColorRecommendation}")
                    div.column
                      i.circle.icon(v-bind:style="{color: sugarsColorRecommendation}")
              
  div.pusher(v-else-if="page==='profile'")
    .ui.equal.width.grid
      .row
        div.ui.cards#healthInfo
          div.card
            div.content
              select.ui.dropdown(v-model="genderInput")
                option#gender(disabled selected value="") Gender
                option(value="male") Male
                option(value="female") Female
          div.card
            div.content
              div.header Age
                div.ui.input
                  input(type='text' placeholder="0" v-model="ageInput")
          div.card#activity
            div.content
              div.header Activity
              select.ui.dropdown(v-model="activityInput")
                option(disabled selected value="") Activity
                option(value="bmr") Basal Metabolic Rate (BMR)
                option(value="sedentary") Sedentary - little or no exercise
                option(value="lightlyActive") Light active - exercise/soprts (1-3 times/week)
                option(value="moderatelyActive") Moderately active - exercise/sports (3-5 times/week)
                option(value="veryActive") Very active - exercise/sports (6-7 times/week)
                option(value="extraActive") Extra active - very hard exercise/sports (twice/day)
      .row
        div.ui.cards#healthInfo
          div.card
            div.content
              div.header Height (cm)
              div.ui.input
                input(type='text' placeholder="0" v-model="heightInput")
          div.card
            div.content
              div.header Weight (kg)
              div.ui.input
                input(type='text' placeholder="0" v-model="weightInput")
          div.card
            div.content
              div.header BMI
              div#bmi {{ bmi }}
        button.huge.ui.button(@click='calculate_calories') Calculate
          .br Calories
      .row
        table.ui.celled.table
          tbody
            td(data-label="category") To maintain your weight you need
            td(data-label="calories") {{ to_maintain }} Kcal/day
          tbody
            td(data-label="category") To lose 0.5 kg per week you need
            td(data-label="calories") {{ to_lose_half_kg }} Kcal/day
          tbody
            td(data-label="category") To lose 1 kg per week you need
            td(data-label="calories") {{ to_lose_a_kg }} Kcal/day
</template>


<script>

import axios from "axios";
import "semantic-ui-offline/semantic.min.css";
import "jquery";
import $ from "jquery";

export default {
  beforeDestory() {
    document.removeEventListener("keyup", this.key);
  },

  created() {
    axios.get("/getImages").then(response => {
      var img_list = JSON.parse(response.data.replace(/'/g, '"'));
      var count = 0;
      var tmp_img_list = [];
      console.log(img_list.images);

      for(var img of img_list.images){
        tmp_img_list.push({'url':'./media/img/' + img + "?t="});
        count++
      }
      this.imgList = tmp_img_list;
      this.numberOfImages = count;
    });
    document.addEventListener("keyup", this.key);
  },

  data() {
    return {
      title: "Inverse Cooking",
      recipe: 100,
      numberOfImages: 0,
      foodName: 'Peanut butter chocolate chip cookies',
      foodIngredient: "2 cup peanut butter, 2 cup sugar, 1 cup brown sugar, 2 whole egg, 2 teaspoon vanilla, 1 all - purpose flour, 1/2 cup cocoa, 1/4 teaspoon salt, 1/4 teaspoon baking soda, 1 semi - sweet chocolate chip, 1 chocolate chip",
      fatInput: "orange",
      saltInput: "green",
      saturatesInput: "orange",
      sugarsInput: "green",
      foodNameRecommendation: 'Zucchini Bread - Bread Machine',
      foodUrlRecommendation: 'http://www.food.com/recipe/zucchini-bread-bread-machine-409757',
      fatInputRecommendation: 'green',
      saltInputRecommendation: 'green',
      saturatesInputRecommendation: 'green',
      sugarsInputRecommendation: 'green',
      page: "home",
      file: "",
      uploadPercentage: 0,
      imgList: "",
      upload_status: "waiting",
      genderInput: '',
      ageInput: '',
      activityInput: '',
      heightInput: '',
      weightInput: '',
      calories: '',
      output: 0
    };
  },
  computed: {
    fatColor() {
      var color = this.fatInput;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    saltColor() {
      var color = this.saltInput;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    saturatesColor() {
      var color = this.saturatesInput;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    sugarsColor() {
      var color = this.sugarsInput;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    fatColorRecommendation() {
      var color = this.fatInputRecommendation;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    saltColorRecommendation() {
      var color = this.saltInputRecommendation;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    saturatesColorRecommendation() {
      var color = this.saturatesInputRecommendation;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    sugarsColorRecommendation() {
      var color = this.sugarsInputRecommendation;
      if (color == 'green')
        return "#14B259";
      else if (color == 'orange')
        return "#FFED51";
      else if (color == 'red')
        return "#E8142A";
      else return "white";
    },
    bmi() {
      var weight = this.weightInput;
      var height = this.heightInput;
      var formula = weight/(height*height)*10000;
      formula = formula.toFixed(2);
      return formula;
    },
    to_maintain() {
      var calories = 0, bmr = 0
      var gender = this.genderInput;
      var age = this.ageInput;
      var activity = this.activityInput;
      var height = this.heightInput;
      var weight = this.weightInput;

      if (gender == 'male')
        bmr = 88.362 + (13.397*weight) + (4.799*height) - (5.677*age);
      else if (gender == 'female')
        bmr = 447.593 + (9.247*weight) + (3.098*height) - (4.330*age);
      else bmr = 0;
      
      if (activity == 'bmr')
        calories = bmr;
      else if (activity == 'sedentary')
        calories = 1.2*bmr;
      else if (activity == "lightlyActive")
        calories = 1.375*bmr;
      else if (activity == "moderatelyActive")
        calories = 1.55*bmr;
      else if (activity =="veryActive" )
        calories = 1.725*bmr;
      else if (activity == "extraActive")
        calories = 1.9*bmr;
      else calories = 0;

      this.calories = calories;
      return calories.toFixed(2);
    },
    to_lose_half_kg() {
      var calories = this.calories - 500;
      return calories.toFixed(2);
    },
    to_lose_a_kg() {
      var calories = this.calories - 1000;
      return calories.toFixed(2);
    }
  },
  methods: {
    submitFile() {
      let formData = new FormData();
      formData.append("image", this.file); //required
      console.log(formData.get("image"));

      axios
        .post("/upload", formData, {
          headers: { "X-CSRFToken": this.getCookie("csrftoken") },
          onUploadProgress: function(progressEvent) {
            this.uploadPercentage = parseInt(
              Math.round((progressEvent.loaded * 100) / progressEvent.total)
            );
          }.bind(this)
        })
        .then(response => {
          console.log(response);
          if (response.status == "200") {
            this.upload_status = "success";
            console.log("success upload!");
          }
        });
    },
    handleFileUpload(event) {
      this.file = this.$refs.file.files[0];
      var name = this.file.name;
      if (this.file) {
        var reader = new FileReader();
        reader.onload = function(e) {
          $(".image-upload-wrap").hide();
          $(".file-upload-image").attr("src", e.target.result);
          $(".file-upload-content").show();
          $(".image-title").html(name);
        };
        reader.readAsDataURL(this.file);
      } else {
        console.log("pic name is empty !");
        removeUpload();
      }
    },
    addImage: function() {
      $(".file-upload-input").trigger("click");
    },
    getCookie(name) {
      var value = "; " + document.cookie;
      var parts = value.split("; " + name + "=");
      if (parts.length === 2)
        return parts
          .pop()
          .split(";")
          .shift();
    },
    show_upload() {
      this.page = "upload";
    },
    show_home() {
      this.page = "home";
      axios.get("/getImages").then(response => {

      var img_list = JSON.parse(response.data.replace(/'/g, '"'));
      var count = 0;
      var tmp_img_list = []
      console.log(img_list.images);
      for(var img of img_list.images){
        tmp_img_list.push({'url':'./media/img/' + img + "?t="});
        count++
      }
      this.imgList = tmp_img_list;
      this.numberOfImages = count;
      this.upload_status = "waiting";
    });
    },
    show_profile() {
      this.page = "profile";
    },
    calculate_calories() {
      var calories = 0, bmr = 0
      var gender = this.genderInput;
      var age = this.ageInput;
      var activity = this.activityInput;
      var height = this.heightInput;
      var weight = this.weightInput;

      if (gender == 'male')
        bmr = 88.362 + (13.397*weight) + (4.799*height) - (5.677*age);
      else if (gender == 'female')
        bmr = 447.593 + (9.247*weight) + (3.098*height) - (4.330*age);
      else bmr = 0;
      
      if (activity == 'bmr')
        calories = bmr;
      else if (activity == 'sedentary')
        calories = 1.2*bmr;
      else if (activity == "lightlyActive")
        calories = 1.375*bmr;
      else if (activity == "moderatelyActive")
        calories = 1.55*bmr;
      else if (activity =="veryActive" )
        calories = 1.725*bmr;
      else if (activity == "extraActive")
        calories = 1.9*bmr;
      else calories = 0;

      this.calories = calories;
    },
    removeUpload : function () {
      $('.file-upload-content').hide();
      $('.image-upload-wrap').show();
      $('.file-upload-input').show()
      
    }
  }
};
</script>

<style lang="sass">
  
.ui.visible.sidebar.left.inverted.vertical.menu
  background-color: rgba(19, 163, 165, 0.8)

h1
  color : #F2F2EF

.ui.item
  color: #F2F2EF
  font-family: 'Dosis', sans-serif
  font-size: 20px
  font-weight: 700

h2
  color : #403833
  font-family: 'Dosis', sans-serif
  font-weight: 500
  text-align: center

h3
  color : #403833
  text-align: center 

.title
  padding: 20px 0 0 0
  text-align: center 

#title_text
  font-family: 'Kalam', cursive
  margin: 5px 0

body
  background-attachment: fixed
  background-image: url(https://images.unsplash.com/photo-1472393365320-db77a5abbecc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=2850&q=80)
  background-size: cover
  height: 100vh
  width: 100vw  

.ui.link.card
  background-color: rgba(242, 242, 239, 0.9)
  border: 1px solid rgba(64, 56, 51, 0.2)
  border-radius: 0
  padding: 15px 15px 60px 15px

.ui.link.card:hover
  background-color: #F2F2EF
  border: 1px solid rgba(64, 56, 51, 0.2)

.file-upload
  background-color: #262626
  margin: 20px auto
  width: 600px
  
.file-upload-btn
  background: #13A3A5
  border: none
  border-radius: 4px
  color: #F2F2EF
  font-weight: 700
  margin: 0
  outline: none
  padding: 10px
  text-transform: uppercase
  transition: all .2s ease
  width: 100%
  
.file-upload-btn:hover
  background: #13A3A5
  color: #403833
  cursor: pointer
  transition: all .2s ease
  
.file-upload-btn:active
  border: 0
  transition: all .2s ease

.file-upload-content
  display: none
  text-align: center

.file-upload-input
  cursor: pointer
  font-color: #262626
  height: 100%
  margin: 0
  opacity: 0
  outline: none
  padding: 0
  position: absolute
  width: 100%
  
.image-upload-wrap
  background: rgba(242, 242, 239, 0.7)
  border: 2px dashed #403833
  margin-top: 20px
  position: relative

.image-dropping,.image-upload-wrap:hover
  background-color: #F2F2EF
  border: 2px dashed #403833

.image-title-wrap
  color: #222
  font-size: 40px
  padding: 0 15px 15px 15px

.image-title
  color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 25px

.drag-text
  text-align: center

.drag-text h3
  color: #403833
  font-family: 'Dosis', sans-serif
  padding: 60px 0
  text-transform: uppercase

.file-upload-image
  margin: auto
  max-height: 200px
  max-width: 200px
  padding: 20px

#foodDetail
  background: rgba(242, 242, 239, 0.8)
  margin: 0 auto
  width: 80vw

#foodDetail column
  padding: 0
  text-align: center

#foodDetail h3
  color: #13A3A5
  font-family: 'Raleway', sans-serif
  font-size: 25px

#foodDetail p, #foodDetail a
  color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 20px
  text-align: center

#foodDetail #foodIngredient
  text-align: left

#foodDetail i.circle.icon
  display: block
  font-size: 25px
  margin: 0 auto
  width: 100%

#healthInfo .card .header
  color: #E54E45
  font-family: 'Dosis', sans-serif
  font-size: 25px
  font-weight: 600
  text-align: center

#healthInfo .card .header i
  font-size: 15px
  margin-left: 5px

#healthInfo .card select
  background: transparent
  border: none
  font-color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 20px
  height: 30px
  margin-left: 30%
  margin-right: 30%
  margin-top: 20px
  padding: 0
  width: 40%

#healthInfo .card select #gender
  color: #E54E45

#healthInfo .card .input
  margin-left: 20%
  margin-right: 20%
  width: 60%

#healthInfo .card .input input
  background: #F2F2EF
  border: none
  color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 25px
  padding-top: 10px
  text-align: center

#healthInfo #activity select
  margin-left: 0
  margin-top: 10px
  width: 100%

#bmi
  color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 25px
  padding-top: 10px
  text-align: center

.huge.ui.button
  background: rgba(19, 163, 165, 0.7)
  border: 2px solid #13A3A5
  font-family: 'Dosis', sans-serif
  font-size: 20px
  height: 80px
  margin-left: 0.8em
  margin-top: 30px
  text-color: #403833

.huge.ui.button:hover
  background: rgba(19, 163, 165, 0.9)

.row .ui.celled.table
  margin-left: 30px
  width: 65%

.ui.table td
  color: #403833
  font-family: 'Raleway', sans-serif
  font-size: 20px
  
</style>

