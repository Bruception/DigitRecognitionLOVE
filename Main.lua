
math.randomseed(os.time())

local NeuralNetwork = require("NeuralNetwork")

local lg = love.graphics
local linid = love.image.newImageData
local lfgd = love.filesystem.getDirectoryItems
local floor, pow = math.floor, math.pow

local TE_DIR = "testing/"
local TR_DIR = "training/"
local CURR_DIR = TR_DIR

local TR_DIGITS = {}
local TE_DIGITS = {}
local D_DIR = {}

local input = {}

function love.load()
  lg.setBackgroundColor(0.172, 0.243, 0.313)
  love.window.setTitle("Digit Recognition")
  for i = 0, 9 do
    TR_DIGITS[i] = lfgd(TR_DIR .. i .. "/")
    TE_DIGITS[i] = lfgd(TE_DIR .. i .. "/")
    D_DIR[i] = (i .. "/")
  end

  for i = 0, 783 do
    input[i] = 0
  end
end

local function findMaxIndex(input)
  local maxIndex = 1
  local max = input:get(1, 1)

  for i = 1, 10 do
    if(input:get(1, i) > max) then
      maxIndex = i
      max = input:get(1, i)
    end
  end

  return maxIndex - 1
end

local myNN = NeuralNetwork:new()
local sample = TR_DIGITS

local timer = 1
local interval = 2

local n = 1
local digit = nil
local digitImage = nil

local expected = {}
local prediction = 0
local output = 0
local error = 0

local totalError = 0
local iterations = 0

function love.update(dt)
  timer = timer + dt

  if(timer >= interval) then

    -- Pick a random digit
    n = math.random(0, 9)

    -- Setup the expected output based on the random digit
    for i = 1, 10 do
      expected[i] = (n == (i - 1)) and 1 or 0
    end

    -- Garbage collection
    digit = nil
    digitImage = nil

    -- Create new image data based on the current source directory
    digit = linid(CURR_DIR .. D_DIR[n] .. sample[n][math.random(#sample[n])])
    digitImage = lg.newImage(digit)

    for i = 0, 27 do
      for j = 0, 27 do
        input[i * 28 + j] = digit:getPixel(i, j) -- Setup the input array
      end
    end

    -- Feedforward and get the output and error
    output, error = myNN:feedforward(input, expected)
    totalError = totalError + error -- Average error sum
    prediction = findMaxIndex(output) -- Find the highest value in the output

    myNN:backpropagate(expected) -- Propagate the error

    iterations = iterations + 1
    timer = 0
  end
end


function love.keypressed(key, scancode, isrepeat)
  if(key == "w") then
    interval = interval + .2
  elseif(key == "s") then
    interval = interval - .2
  end

  if(key == "t") then
    CURR_DIR = (CURR_DIR == TR_DIR) and TE_DIR or TR_DIR
    sample = (CURR_DIR == TE_DIR) and TE_DIGITS or TR_DIGITS
  end
end

local function round(x, places)
  local mult = pow(10, places or 3)
  return floor(x * mult) / mult
end

function love.draw()
  if(digitImage ~= nil) then
    lg.draw(digitImage, 175, 240, 0, 2, 2)
  end

  myNN:draw()

  lg.setColor(1, 1, 1)
  lg.print(CURR_DIR, 500, 75)

  local roundAvgError = round(totalError / iterations)
  local roundError = round(error)

  lg.print("Average Error: " .. roundAvgError * 100 .. "%", 500, 175)
  lg.print("Average Accuracy: " .. (1 - roundAvgError) * 100 .. "%", 500, 200)

  lg.print("This is a .. " .. prediction, 500, 265, 0, 2, 2)

  lg.print("Error: " .. roundError * 100 .. "%", 500, 360)
  lg.print("Accuracy: " .. (1 - roundError) * 100 .. "%", 500, 375)
end
