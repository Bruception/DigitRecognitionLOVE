local max, abs = math.max, math.abs
local Matrix = require("Matrix")

local lg = love.graphics

local NeuralNetwork = {}

function NeuralNetwork:init()
  ---------- LAYERS ------------
  self.input = Matrix:new(1, 784)

  self.h = Matrix:new(784, 23)
  self.hBias = Matrix:new(1, 23)

  self.h:randomize()
  self.hBias:randomize()

  self.h2 = Matrix:new(23, 23)
  self.h2Bias = Matrix:new(1, 23)

  self.h2:randomize()
  self.h2Bias:randomize()

  self.o = Matrix:new(23, 10)
  self.oBias = Matrix:new(1, 10)

  self.o:randomize()
  self.oBias:randomize()

  ------- OUTPUTS OF LAYERS ----------
  self.hOut = nil
  self.oOut = nil
  self.h2Out = nil

  ------- HYPER PARAMETER(S) --------
  self.lr = -0.1

  return self
end

local function sigmoid(x)
  local sum = 1 + math.exp(-x)

  return 1 / sum
end

local function MSError(output, expected)
  local sum = 0

  local diff = 0

  for i = 1, 10 do
    diff = (expected[i] - output:get(1, i))
    sum = sum + (diff * diff)
  end

  return sum / 10
end

-- Feedforward can be improved
function NeuralNetwork:feedforward(pixels, expected)

  for i = 0, 783 do
    self.input:set(1, i + 1, pixels[i])
  end

  local hOut = self.input * self.h
  hOut = hOut + self.hBias
  hOut = hOut:apply(sigmoid)
  self.hOut = hOut

  local h2Out = self.hOut * self.h2
  h2Out = h2Out + self.h2Bias
  h2Out = h2Out:apply(sigmoid)
  self.h2Out = h2Out

  local oOut = self.h2Out * self.o
  oOut = oOut + self.oBias
  oOut = oOut:apply(sigmoid)
  self.oOut = oOut

  local error = MSError(self.oOut, expected)

  return oOut, error
end

-- This implementation can be improved.
-- Need to study more calculus and linear algebra :P
function NeuralNetwork:backpropagate(expected)
  local expectedMatrix = Matrix:new(1, 10)

  for i = 1, 10 do
    expectedMatrix:set(1, i, expected[i])
  end

  -- Output layer -> Hidden Layer 2 (Output Weights)
  local dE_dOut = (expectedMatrix - self.oOut) * -2
  local dOut_dWSO = self.oOut:eDot((self.oOut * -1) + 1);
  local dWSO_dW = self.h2Out
  local dE_dOW = (dE_dOut:eDot(dOut_dWSO)):transpose();

  self.o = self.o + ((dE_dOW * dWSO_dW) * self.lr):transpose()
  self.oBias = self.oBias + (dE_dOW * self.lr):transpose()

  -- Hiden Layer 2 -> Hidden Layer 1 (Hidden 2 Weights)
  local dOut_dWSH2 = self.h2Out:eDot((self.h2Out * -1) + 1);
  local dE_dH2W = (dE_dOut:eDot(dOut_dWSO)) * self.o:transpose()
  dE_dH2W = dE_dH2W:eDot(dOut_dWSH2):transpose()

  self.h2 = self.h2 + ((dE_dH2W * self.hOut) * self.lr):transpose()
  self.h2Bias = self.h2Bias + (dE_dH2W * self.lr):transpose()

  -- Hidden Layer 1 -> Input (Hidden 1 Weights)
  local dOut_dWSH = self.hOut:eDot((self.hOut * -1) + 1)
  local dE_dHW = (dE_dOut:eDot(dOut_dWSO)) * self.o:transpose()
  dE_dHW = dE_dHW:eDot(dOut_dWSH2) * self.h2:transpose()
  dE_dHW = dE_dHW:eDot(dOut_dWSH):transpose()

  self.h = self.h + ((dE_dHW * self.input) * self.lr):transpose()
  self.hBias = self.hBias + (dE_dHW * self.lr):transpose()
end


-------- DRAW CODE -------------
local neuronSize = 7
local paddingY = 20
local paddingX = 60
local currentWeight = 0
local offsetX = 200
local offsetY = 100

local r, g, b = 0, 0, 0

local function drawWeights(layer, layer2, ln, oy, oy2)

  for i = 1, layer2.n do
    for j = 1, layer2.m do

      currentWeight = layer2:get(i, j)

      if(currentWeight > 0) then
        r, g, b = 0.16, 0.501, 0.725
      else
        r, g, b = 0.752, 0.223, 0.168
      end

      lg.setColor(r, g, b)
      lg.setLineWidth(abs(currentWeight))
      lg.line(offsetX + (paddingX * (ln + 1)), offsetY + (j * paddingY) + oy2, offsetX + (paddingX * ln), offsetY + (i * paddingY) + oy)
    end
  end
end

local currenAct

local function drawNeurons(neurons, ln, activation, oy)
  for i = 1, neurons.m do
    currenAct = 1

    if(activation ~= nil) then
      currenAct = activation:get(1, i)
    end

    lg.setColor(currenAct, currenAct, currenAct)
    lg.circle("fill", offsetX + (paddingX * ln), offsetY + (i * paddingY) + oy, neuronSize)
  end
end

function NeuralNetwork:draw()

  -- Input layer Edges
  --drawWeights(self.input, self.h, 1, -7830, -50)
  drawWeights(self.h, self.h2, 2, -50, -50)
  drawWeights(self.h2, self.o, 3, -50, 80)

  lg.setLineWidth(1)

  -- Input layer neurons
  --drawNeurons(self.input, 1, self.input, -7830)
  drawNeurons(self.h, 2, self.hOut, -50)
  drawNeurons(self.h2, 3, self.h2Out, -50)
  drawNeurons(self.o, 4, self.oOut, 80)

end

return NeuralNetwork:init()
